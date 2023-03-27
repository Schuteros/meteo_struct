use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use serde_derive::{Deserialize, Serialize};
use csv;

#[derive(Debug, Deserialize, Serialize)]
struct Record {
    STATION_ID: String,
    ABBREVIATION: String,
    DATETIME: String,
    VALUE: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path("meteorologiskie_arhiva_dati.csv")?;
    let mut records = Vec::new();
    for result in rdr.deserialize() {
        let record: Record = result?;
        if &record.STATION_ID == "RIGASLU" {
            records.push(record);
        }
    }
    println!("{:?}", records);
    Ok(())
}