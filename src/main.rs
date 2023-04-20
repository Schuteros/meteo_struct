use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use serde_derive::{Deserialize, Serialize};
use csv;
use csv::{WriterBuilder, Writer};
use std::collections::BTreeMap;
use std::ops::Bound::Included;
use chrono::{DateTime, Utc, TimeZone};

#[derive(Debug, Deserialize, Serialize)]
struct Record {
    STATION_ID: String,
    ABBREVIATION: String,
    DATETIME: String,
    VALUE: f32,
}

#[derive(Debug, Deserialize, Serialize)]
struct FullRecord {
    TDRY: f32,
    RLH: f32,
    PRSS: f32,
    HPRAB: f32,
}

#[derive(Debug, Deserialize, Serialize)]
struct Output {
    datetime: i64,
    TDRY: f32,
    RLH: f32,
    PRSS: f32,
    HPRAB: f32,
}

#[derive(Debug, Deserialize, Serialize)]
struct Data_1 {
    data: String,
}

fn main() {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path("meteorologiskie_arhiva_dati.csv").unwrap();
    let mut records = Vec::new();
    for result in rdr.deserialize() {
        let record: Record = result.unwrap();
        if &record.STATION_ID == "RIGASLU" {
            records.push(record);
        }
    }
    println!("{:?}", &records);

    let mut sorted_records: BTreeMap<String, FullRecord> = BTreeMap::new();

    for record in records {
        if sorted_records.contains_key(&record.DATETIME) {
            if &record.ABBREVIATION == "TDRY" {
                let sort_record = sorted_records.get_mut(&record.DATETIME).unwrap();
                sort_record.TDRY = record.VALUE;
            } else if &record.ABBREVIATION == "RLH" {
                let sort_record = sorted_records.get_mut(&record.DATETIME).unwrap();
                sort_record.RLH = record.VALUE;
            } else if &record.ABBREVIATION == "PRSS" {
                let sort_record = sorted_records.get_mut(&record.DATETIME).unwrap();
                sort_record.PRSS = record.VALUE;
            } else if &record.ABBREVIATION == "HPRAB" {
                let sort_record = sorted_records.get_mut(&record.DATETIME).unwrap();
                sort_record.HPRAB = record.VALUE;
            }
        } else {
            sorted_records.insert(record.DATETIME.clone(), FullRecord { TDRY: 0f32, RLH: 0f32, PRSS: 0f32, HPRAB: 0f32 });

            if &record.ABBREVIATION == "TDRY" {
                let sort_record = sorted_records.get_mut(&record.DATETIME).unwrap();
                sort_record.TDRY = record.VALUE;
            } else if &record.ABBREVIATION == "RLH" {
                let sort_record = sorted_records.get_mut(&record.DATETIME).unwrap();
                sort_record.RLH = record.VALUE;
            } else if &record.ABBREVIATION == "PRSS" {
                let sort_record = sorted_records.get_mut(&record.DATETIME).unwrap();
                sort_record.PRSS = record.VALUE;
            } else if &record.ABBREVIATION == "HPRAB" {
                let sort_record = sorted_records.get_mut(&record.DATETIME).unwrap();
                sort_record.HPRAB = record.VALUE;
            }
        }
    }
    println!("{:?}", &sorted_records);

    let mut sorted_records_datetime_numeric: BTreeMap<i64, FullRecord> = BTreeMap::new();

    for record in sorted_records {
        let dt = Utc.datetime_from_str(&record.0, "%Y.%m.%d %H:%M:%S").unwrap();
        sorted_records_datetime_numeric.insert(dt.timestamp(), record.1);
    }

    let mut sorted_records: BTreeMap<i64, FullRecord> = BTreeMap::new();

    while !&sorted_records_datetime_numeric.is_empty() {
        let  mut smallest = sorted_records_datetime_numeric.first_key_value().unwrap().0.clone();
        for value in &sorted_records_datetime_numeric {
            if value.0 < &smallest {
                smallest = value.0.clone();
            }
        }
        let full_record = sorted_records_datetime_numeric.get(&smallest).unwrap();
        let full_record = FullRecord {
            TDRY: full_record.TDRY,
            RLH: full_record.RLH,
            PRSS: full_record.PRSS,
            HPRAB: full_record.HPRAB,
        };

        sorted_records.insert(smallest, full_record);
        sorted_records_datetime_numeric.remove(&smallest);
    }

    let mut data_1: Vec<Vec<f32>> = Vec::new();

    let length = sorted_records.len();
    let mut i = 0;
    for data in &sorted_records {
        if length - 24 >= i {
            let mut element = Vec::new();
            for record in sorted_records.range((Included(*data.0), Included(data.0 + 3600*24+1))) {
                element.push(record.1.TDRY);
                element.push(record.1.RLH);
                element.push(record.1.PRSS);
                element.push(record.1.HPRAB);

            }
            data_1.push(element);
            i+=1;
        }
    }
    println!("{:?}", data_1);

    let mut writer = WriterBuilder::new().from_path("output.csv").unwrap();
    for record in data_1 {
        let mut data = String::new();
        for value in record {
            data = data + &value.to_string() + " ";
        }
        data = data
            .trim()
            .replace(" ", ",");
        let row = Data_1 {
            data,
        };
        writer.serialize(row).unwrap();
    }
    writer.flush().unwrap();
}