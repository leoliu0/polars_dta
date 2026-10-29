//! Mapping Arrow columns to Stata variables: storage type, display format and value label.

use std::collections::HashMap;

use arrow_arith::aggregate::{max, min};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, ArrowNumericType, LargeStringArray, RecordBatch, StringArray, StringViewArray,
};
use arrow_schema::{DataType, Schema, TimeUnit};

use crate::error::{Error, Result};

/// Days from 1960-01-01, Stata's epoch, to 1970-01-01, Arrow's.
pub const EPOCH_DAYS: i64 = 3653;
pub const EPOCH_MS: i64 = EPOCH_DAYS * 86_400_000;

/// Nonmissing ranges; larger values are Stata's missing codes.
const BYTE_RANGE: (i128, i128) = (-127, 100);
const INT_RANGE: (i128, i128) = (-32_767, 32_740);
const LONG_RANGE: (i128, i128) = (-2_147_483_647, 2_147_483_620);
/// Integers a double holds exactly.
const DOUBLE_INT_RANGE: (i128, i128) = (-(1 << 53), 1 << 53);
/// Largest nonmissing float and double.
const FLOAT_MAX: f32 = f32::from_bits(0x7eff_ffff);
const DOUBLE_MAX: f64 = f64::from_bits(0x7fdf_ffff_ffff_ffff);

pub const STR_MAX: usize = 2045;
const STRL_MAX: usize = 2_000_000_000;
const MAX_VARS_118: usize = 32_767;
const NAME_MAX_CHARS: usize = 32;
const LABEL_MAX_CHARS: usize = 80;
const VALUE_LABEL_TEXT_MAX: usize = 32_000;
const VALUE_LABEL_MAX_CODES: usize = 65_536;

const RESERVED: [&str; 20] = [
    "_all", "_b", "byte", "_coef", "_cons", "double", "float", "if", "in", "int", "long", "_n",
    "_N", "_pi", "_pred", "_rc", "_skip", "strL", "using", "with",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Storage {
    Byte,
    Int,
    Long,
    Float,
    Double,
    Str(u16),
    StrL,
}

impl Storage {
    pub fn width(self) -> usize {
        match self {
            Storage::Byte => 1,
            Storage::Int => 2,
            Storage::Long | Storage::Float => 4,
            Storage::Double | Storage::StrL => 8,
            Storage::Str(w) => w as usize,
        }
    }

    /// Type code in releases 117-119.
    pub fn code(self) -> u16 {
        match self {
            Storage::Str(w) => w,
            Storage::StrL => 32_768,
            Storage::Double => 65_526,
            Storage::Float => 65_527,
            Storage::Long => 65_528,
            Storage::Int => 65_529,
            Storage::Byte => 65_530,
        }
    }

    /// The format Stata gives a new variable of this type.
    fn default_format(self) -> String {
        match self {
            Storage::Byte | Storage::Int => "%8.0g".into(),
            Storage::Long => "%12.0g".into(),
            Storage::Float => "%9.0g".into(),
            Storage::Double => "%10.0g".into(),
            Storage::Str(w) => format!("%{}s", w.max(9)),
            Storage::StrL => "%9s".into(),
        }
    }
}

/// How cells are read from the Arrow arrays of a column.
#[derive(Debug)]
pub enum Source {
    Bool,
    Int,
    Float,
    Str,
    /// Dictionary-encoded strings; per batch, the Stata code of each dictionary entry.
    Dict(Vec<Vec<i32>>),
    /// Days since 1970.
    Date,
    /// Since 1970, without time zone.
    Timestamp(TimeUnit),
    /// Since midnight.
    Time(TimeUnit),
}

#[derive(Debug)]
pub struct Column {
    pub name: String,
    pub storage: Storage,
    pub source: Source,
    pub format: String,
    pub value_label: Option<String>,
    pub label: String,
}

/// A value-label definition, entries in ascending value order.
#[derive(Debug)]
pub struct ValueLabel {
    pub name: String,
    pub entries: Vec<(i32, Vec<u8>)>,
}

#[derive(Debug)]
pub struct Plan {
    pub release: u16,
    pub nobs: u64,
    pub data_label: String,
    pub timestamp: String,
    pub columns: Vec<Column>,
    pub value_labels: Vec<ValueLabel>,
}

#[derive(Debug, Default)]
pub struct Options {
    pub variable_labels: HashMap<String, String>,
    pub value_labels: HashMap<String, HashMap<i64, String>>,
    pub data_label: String,
    pub timestamp: String,
}

/// String cells of any Arrow string layout.
pub enum Strings<'a> {
    Utf8(&'a StringArray),
    Large(&'a LargeStringArray),
    View(&'a StringViewArray),
}

impl<'a> Strings<'a> {
    pub fn new(a: &'a dyn Array) -> Option<Self> {
        match a.data_type() {
            DataType::Utf8 => Some(Strings::Utf8(a.as_string())),
            DataType::LargeUtf8 => Some(Strings::Large(a.as_string())),
            DataType::Utf8View => Some(Strings::View(a.as_string_view())),
            _ => None,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Strings::Utf8(a) => a.len(),
            Strings::Large(a) => a.len(),
            Strings::View(a) => a.len(),
        }
    }

    #[inline]
    pub fn get(&self, i: usize) -> Option<&'a [u8]> {
        match self {
            Strings::Utf8(a) => a.is_valid(i).then(|| a.value(i).as_bytes()),
            Strings::Large(a) => a.is_valid(i).then(|| a.value(i).as_bytes()),
            Strings::View(a) => a.is_valid(i).then(|| a.value(i).as_bytes()),
        }
    }
}

pub fn plan(schema: &Schema, batches: &[RecordBatch], opts: &Options) -> Result<Plan> {
    let k = schema.fields().len();
    let release = if k > MAX_VARS_118 { 119 } else { 118 };
    let nobs: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
    let mut columns = Vec::with_capacity(k);
    let mut value_labels = Vec::new();

    for (j, field) in schema.fields().iter().enumerate() {
        let name = field.name();
        check_name(name)?;
        let arrays: Vec<&ArrayRef> = batches.iter().map(|b| b.column(j)).collect();
        let err = |msg: String| Error::invalid(format!("column {name:?}: {msg}"));
        let mut value_label = None;
        let (storage, source, format) = match field.data_type() {
            DataType::Boolean => (Storage::Byte, Source::Bool, None),
            DataType::Int8 => int_column::<Int8Type>(&arrays, Storage::Byte).map_err(err)?,
            DataType::Int16 => int_column::<Int16Type>(&arrays, Storage::Int).map_err(err)?,
            DataType::UInt8 => int_column::<UInt8Type>(&arrays, Storage::Int).map_err(err)?,
            DataType::Int32 => int_column::<Int32Type>(&arrays, Storage::Long).map_err(err)?,
            DataType::UInt16 => int_column::<UInt16Type>(&arrays, Storage::Long).map_err(err)?,
            DataType::Int64 => int_column::<Int64Type>(&arrays, Storage::Long).map_err(err)?,
            DataType::UInt32 => int_column::<UInt32Type>(&arrays, Storage::Long).map_err(err)?,
            DataType::UInt64 => int_column::<UInt64Type>(&arrays, Storage::Long).map_err(err)?,
            DataType::Float32 => {
                let mut wide = false;
                for v in arrays
                    .iter()
                    .flat_map(|a| a.as_primitive::<Float32Type>().iter().flatten())
                {
                    if v.is_infinite() {
                        return Err(err("contains infinity, which Stata cannot store".into()));
                    }
                    wide |= v.abs() > FLOAT_MAX;
                }
                let storage = if wide {
                    Storage::Double
                } else {
                    Storage::Float
                };
                (storage, Source::Float, None)
            }
            DataType::Float64 => {
                for v in arrays
                    .iter()
                    .flat_map(|a| a.as_primitive::<Float64Type>().iter().flatten())
                {
                    if v.is_infinite() {
                        return Err(err("contains infinity, which Stata cannot store".into()));
                    }
                    if v.abs() > DOUBLE_MAX {
                        return Err(err(format!("value {v} exceeds Stata's largest double")));
                    }
                }
                (Storage::Double, Source::Float, None)
            }
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                (string_storage(&arrays).map_err(err)?, Source::Str, None)
            }
            DataType::Dictionary(_, values)
                if matches!(
                    **values,
                    DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
                ) =>
            {
                let (storage, remaps, entries) = dictionary_column(&arrays).map_err(err)?;
                value_labels.push(ValueLabel {
                    name: name.clone(),
                    entries,
                });
                value_label = Some(name.clone());
                (storage, Source::Dict(remaps), None)
            }
            DataType::Date32 => {
                for d in arrays
                    .iter()
                    .flat_map(|a| a.as_primitive::<Date32Type>().iter().flatten())
                {
                    if !in_range(d as i128 + EPOCH_DAYS as i128, LONG_RANGE) {
                        return Err(err(format!(
                            "date {d} days after 1970-01-01 is outside Stata's range"
                        )));
                    }
                }
                (Storage::Long, Source::Date, Some("%td"))
            }
            DataType::Timestamp(_, Some(tz)) => {
                return Err(err(format!(
                    "Stata has no time zones; convert the {tz} timestamps to local time first"
                )));
            }
            DataType::Timestamp(unit, None) => {
                (Storage::Double, Source::Timestamp(*unit), Some("%tc"))
            }
            DataType::Time32(unit) | DataType::Time64(unit) => {
                (Storage::Double, Source::Time(*unit), Some("%tcHH:MM:SS"))
            }
            other => {
                return Err(err(format!(
                    "Stata has no type for {other}; cast the column first"
                )));
            }
        };
        columns.push(Column {
            name: name.clone(),
            storage,
            source,
            format: format.map_or_else(|| storage.default_format(), str::to_string),
            value_label,
            label: String::new(),
        });
    }

    if columns.iter().any(|c| c.storage == Storage::StrL) {
        // A strL cell refers to its GSO by (variable, observation) packed into 8 bytes.
        let o_bits = if release == 118 { 48 } else { 40 };
        if nobs >= 1 << o_bits {
            return Err(Error::invalid(format!(
                "{nobs} observations are too many for strL variables in release {release}"
            )));
        }
    }

    let index: HashMap<&str, usize> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(j, f)| (f.name().as_str(), j))
        .collect();
    let find = |what: &str, name: &str| {
        index
            .get(name)
            .copied()
            .ok_or_else(|| Error::invalid(format!("{what}: there is no column {name:?}")))
    };

    let mut variable_labels: Vec<_> = opts.variable_labels.iter().collect();
    variable_labels.sort();
    for (name, label) in variable_labels {
        let j = find("variable_labels", name)?;
        check_label(label, &format!("variable label of {name:?}"))?;
        columns[j].label = label.clone();
    }

    let mut user_value_labels: Vec<_> = opts.value_labels.iter().collect();
    user_value_labels.sort_by_key(|(name, _)| *name);
    for (name, labels) in user_value_labels {
        let j = find("value_labels", name)?;
        if !matches!(columns[j].source, Source::Int | Source::Bool) {
            return Err(Error::invalid(format!(
                "value_labels: column {name:?} is not an integer or boolean column"
            )));
        }
        if labels.len() > VALUE_LABEL_MAX_CODES {
            return Err(Error::invalid(format!(
                "value_labels: {} labels for {name:?} exceed Stata's limit of {VALUE_LABEL_MAX_CODES}",
                labels.len()
            )));
        }
        let mut entries = Vec::with_capacity(labels.len());
        for (&value, text) in labels {
            if !in_range(value as i128, LONG_RANGE) {
                return Err(Error::invalid(format!(
                    "value_labels: value {value} of {name:?} is outside -2,147,483,647..2,147,483,620"
                )));
            }
            check_value_label_text(text.as_bytes()).map_err(|msg| {
                Error::invalid(format!("value_labels: label of {value} for {name:?} {msg}"))
            })?;
            entries.push((value as i32, text.as_bytes().to_vec()));
        }
        entries.sort_unstable_by_key(|&(v, _)| v);
        columns[j].value_label = Some(name.clone());
        value_labels.push(ValueLabel {
            name: name.clone(),
            entries,
        });
    }

    check_label(&opts.data_label, "data label")?;
    if opts.timestamp.len() > u8::MAX as usize || opts.timestamp.contains('\0') {
        return Err(Error::invalid(
            "timestamp must be at most 255 bytes without NUL",
        ));
    }

    Ok(Plan {
        release,
        nobs,
        data_label: opts.data_label.clone(),
        timestamp: opts.timestamp.clone(),
        columns,
        value_labels,
    })
}

fn in_range(v: i128, (lo, hi): (i128, i128)) -> bool {
    lo <= v && v <= hi
}

/// The smallest type from `natural` up that holds every value exactly.
fn int_column<T>(
    arrays: &[&ArrayRef],
    natural: Storage,
) -> Result<(Storage, Source, Option<&'static str>), String>
where
    T: ArrowNumericType,
    T::Native: Into<i128>,
{
    let mut range: Option<(i128, i128)> = None;
    for a in arrays {
        let a = a.as_primitive::<T>();
        if let (Some(lo), Some(hi)) = (min(a), max(a)) {
            let (lo, hi) = (lo.into(), hi.into());
            range = Some(range.map_or((lo, hi), |(l, h)| (l.min(lo), h.max(hi))));
        }
    }
    let Some((lo, hi)) = range else {
        return Ok((natural, Source::Int, None));
    };
    [
        (Storage::Byte, BYTE_RANGE),
        (Storage::Int, INT_RANGE),
        (Storage::Long, LONG_RANGE),
        (Storage::Double, DOUBLE_INT_RANGE),
    ]
    .into_iter()
    .find(|&(s, r)| s >= natural && in_range(lo, r) && in_range(hi, r))
    .map(|(s, _)| (s, Source::Int, None))
    .ok_or_else(|| {
        format!(
            "values from {lo} to {hi} exceed ±2^53, the integers a Stata double holds exactly; \
             cast to Float64 to accept the rounding or to String to keep every digit"
        )
    })
}

fn string_storage(arrays: &[&ArrayRef]) -> Result<Storage, String> {
    let mut width = 0;
    let mut has_nul = false;
    for a in arrays {
        let s = Strings::new(a.as_ref()).expect("string array");
        for v in (0..s.len()).filter_map(|i| s.get(i)) {
            width = width.max(v.len());
            has_nul |= v.contains(&0);
        }
    }
    if width > STRL_MAX {
        return Err(format!(
            "a {width}-byte string exceeds Stata's strL limit of {STRL_MAX} bytes"
        ));
    }
    // str# cells end at the first NUL; strL keeps it.
    Ok(if width > STR_MAX || has_nul {
        Storage::StrL
    } else {
        Storage::Str(width.max(1) as u16)
    })
}

/// Codes 1, 2, ... in dictionary order, unused entries included (an Enum keeps all its
/// categories). Batches may carry different dictionaries; equal strings share a code.
type DictionaryColumn = (Storage, Vec<Vec<i32>>, Vec<(i32, Vec<u8>)>);

fn dictionary_column(arrays: &[&ArrayRef]) -> Result<DictionaryColumn, String> {
    let mut codes: HashMap<&[u8], i32> = HashMap::new();
    let mut entries: Vec<(i32, Vec<u8>)> = Vec::new();
    let mut remaps = Vec::with_capacity(arrays.len());
    for a in arrays {
        let values = Strings::new(a.as_any_dictionary().values().as_ref()).expect("string values");
        let mut remap = Vec::with_capacity(values.len());
        for i in 0..values.len() {
            let text = values.get(i).ok_or("a category is null")?;
            let code = match codes.get(text) {
                Some(&code) => code,
                None => {
                    if entries.len() == VALUE_LABEL_MAX_CODES {
                        return Err(format!(
                            "more than {VALUE_LABEL_MAX_CODES} categories, Stata's limit for a \
                             value label; cast to String instead"
                        ));
                    }
                    check_value_label_text(text).map_err(|msg| format!("a category {msg}"))?;
                    let code = entries.len() as i32 + 1;
                    codes.insert(text, code);
                    entries.push((code, text.to_vec()));
                    code
                }
            };
            remap.push(code);
        }
        remaps.push(remap);
    }
    let n = entries.len() as i128;
    let storage = if n <= BYTE_RANGE.1 {
        Storage::Byte
    } else if n <= INT_RANGE.1 {
        Storage::Int
    } else {
        Storage::Long
    };
    Ok((storage, remaps, entries))
}

fn check_value_label_text(text: &[u8]) -> Result<(), String> {
    if text.len() > VALUE_LABEL_TEXT_MAX {
        Err(format!(
            "is {} bytes, more than Stata's {VALUE_LABEL_TEXT_MAX}",
            text.len()
        ))
    } else if text.contains(&0) {
        Err("contains NUL".into())
    } else {
        Ok(())
    }
}

fn check_label(label: &str, what: &str) -> Result<()> {
    if label.chars().count() > LABEL_MAX_CHARS || label.contains('\0') {
        return Err(Error::invalid(format!(
            "{what} must be at most {LABEL_MAX_CHARS} characters without NUL"
        )));
    }
    Ok(())
}

fn check_name(name: &str) -> Result<()> {
    let mut chars = name.chars();
    let well_formed = match chars.next() {
        Some(c) if c == '_' || c.is_alphabetic() => chars.all(|c| c == '_' || c.is_alphanumeric()),
        _ => false,
    };
    if !well_formed || name.chars().count() > NAME_MAX_CHARS {
        return Err(Error::invalid(format!(
            "{name:?} is not a valid Stata variable name: use 1 to {NAME_MAX_CHARS} letters, \
             digits or underscores, not starting with a digit"
        )));
    }
    let str_type = name
        .strip_prefix("str")
        .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()));
    if str_type || RESERVED.contains(&name) {
        return Err(Error::invalid(format!("{name:?} is reserved in Stata")));
    }
    Ok(())
}
