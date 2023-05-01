use serde_derive::{Deserialize, Serialize};
use csv;
use csv::{WriterBuilder, Writer};
use std::collections::BTreeMap;
use std::ops::Bound::Included;
use chrono::{Utc, TimeZone};

// DenseMatrix wrapper around Vec
use smartcore::linalg::basic::matrix::DenseMatrix;
// SVM
use smartcore::svm::svr::{SVRParameters, SVR};
use smartcore::svm::{Kernels};
// Random Forest
use smartcore::ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters};
// Model performance
use smartcore::model_selection::train_test_split;
use smartcore::metrics::{mean_squared_error, precision};


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

#[derive(Debug, Deserialize, Serialize)]
struct Label {
    labels: i64
}

fn read_data(path: &str) -> BTreeMap<i64, FullRecord> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path(path).unwrap();
    let mut records = Vec::new();
    for result in rdr.deserialize() {
        let record: Record = result.unwrap();
        if &record.STATION_ID == "RIGASLU" {
            records.push(record);
        }
    }

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
    sorted_records
}

fn create_data(records: &BTreeMap<i64, FullRecord>, hours: usize, hours_per_data: i32, after_hours: f32) -> (DenseMatrix<f32>, Vec<f32>) {
    let mut training_data: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();

    let length = records.len();
    let mut i = 0;

    let mut writer = WriterBuilder::new().from_path("output.csv").unwrap();
    let mut label_writer = WriterBuilder::new().from_path("labels.csv").unwrap();

    for data in records {
        let mut string = String::new();
        if length - 390 >= i {
            let mut element = Vec::new();
            let mut j = 0;
            for record in records.range((Included(*data.0), Included(data.0 + 3600 * hours as i64 +1))) {
                if j % hours_per_data == 0 {
                    element.push(record.1.TDRY);
                    element.push(record.1.RLH);
                    element.push(record.1.PRSS);
                    element.push(record.1.HPRAB);
                }
                j+=1
            }
            for value in &element {
                string = string + &value.to_string() + " ";
            }
            training_data.push(element);

            string = string
                .trim()
                .replace(" ", ",");

            labels.push(records.get(&(data.0 + 3600i64* after_hours as i64 + 3600*24)).unwrap().TDRY as f32);

            let row = Data_1 {
                data: string,
            };
            writer.serialize(row).unwrap();
            label_writer.serialize(labels[i]).unwrap();
            i+=1;
        }
    }
    writer.flush().unwrap();
    label_writer.flush().unwrap();
    let x = DenseMatrix::from_2d_array( &training_data.iter().map(|v| v.as_slice()).collect::<Vec<_>>()[..]);
    (x, labels)
}

fn create_data_1(records: &BTreeMap<i64, FullRecord>, hours: usize, hours_per_data: i32, after_hours: f32) -> (DenseMatrix<f32>, Vec<f32>) {
    let mut training_data: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();

    let length = records.len();
    let mut i = 0;

    let mut writer = WriterBuilder::new().from_path("output.csv").unwrap();
    let mut label_writer = WriterBuilder::new().from_path("labels.csv").unwrap();

    for data in records {
        let mut string = String::new();
        if length - 390 >= i {
            let mut element = Vec::new();
            let mut j = 0;
            for record in records.range((Included(*data.0), Included(data.0 + 3600 * hours as i64 +1))) {
                if j % hours_per_data == 0 {
                    element.push(record.1.TDRY);
                    element.push(record.1.RLH);
                    element.push(record.1.PRSS);
                    element.push(record.1.HPRAB);
                }
                j+=1
            }
            for value in &element {
                string = string + &value.to_string() + " ";
            }
            training_data.push(element);

            string = string
                .trim()
                .replace(" ", ",");

            labels.push(records.get(&(data.0 + 3600i64* after_hours as i64 + 3600*24)).unwrap().HPRAB as f32);

            let row = Data_1 {
                data: string,
            };
            writer.serialize(row).unwrap();
            label_writer.serialize(labels[i]).unwrap();
            i+=1;
        }
    }
    writer.flush().unwrap();
    label_writer.flush().unwrap();
    let x = DenseMatrix::from_2d_array( &training_data.iter().map(|v| v.as_slice()).collect::<Vec<_>>()[..]);
    (x, labels)
}


fn main() {
    let sorted_records = read_data("meteorologiskie_arhiva_dati.csv");

    // 24 hour data, predict after 24 hours
    /*
    let (x, y) = create_data(&sorted_records, 24, 5, 24f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.001f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    println!(
        "MSE: {}",
        (mean_squared_error(&y_test, &y_hat_svm)).sqrt()
    );
     */
    // Random Forest
/*
    let (x, y) = create_data(&sorted_records, 24, 2, 24f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(30),
        min_samples_leaf: 1,
        min_samples_split: 1,
        n_trees: 50,
        m: Some(5),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
    // Calculate test error
    println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf).sqrt());
*/
    // 24 hour data predict after 72 hour
/*
    let (x, y) = create_data(&sorted_records, 24, 5, 72f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.005f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    println!(
        "MSE: {}",
        mean_squared_error(&y_test, &y_hat_svm).sqrt()
    );
*/
    // Random Forest
/*
    let (x, y) = create_data(&sorted_records, 24, 2, 72f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(40),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 60,
        m: Some(6),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
    // Calculate test error
    println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf).sqrt());
 */
    // 24 hour data predict after 7 days
/*
    let (x, y) = create_data(&sorted_records, 24, 5, 168f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.005f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    println!(
        "MSE: {}",
        mean_squared_error(&y_test, &y_hat_svm).sqrt()
    );
*/
    // Random Forest
/*
    let (x, y) = create_data(&sorted_records, 24, 1, 168f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(40),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 40,
        m: Some(6),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
    // Calculate test error
    println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf).sqrt());
*/

    // 72 hour data predict after 24 hour
/*
    let (x, y) = create_data(&sorted_records, 72, 12, 24f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.00001f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    println!(
        "MSE: {}",
        mean_squared_error(&y_test, &y_hat_svm).sqrt()
    );
*/
    // Random Forest
/*
    let (x, y) = create_data(&sorted_records, 72, 1, 24f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(30),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 60,
        m: Some(10),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
    // Calculate test error
    println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf).sqrt());
*/
    // 72 hour data predict after 72 hour
/*
    let (x, y) = create_data(&sorted_records, 72, 12, 72f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.001f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    println!(
        "MSE: {}",
        mean_squared_error(&y_test, &y_hat_svm).sqrt()
    );
*/
    // Random Forest
    /*
    let (x, y) = create_data(&sorted_records, 72, 1, 72f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(50),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 60,
        m: Some(6),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
    // Calculate test error
    println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf).sqrt());
*/
    // 72 hour data predict after 7 days
    /*
    let (x, y) = create_data(&sorted_records, 72, 12, 168f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.001f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    println!(
        "MSE: {}",
        mean_squared_error(&y_test, &y_hat_svm).sqrt()
    );
*/
    // Random Forest
    /*
    let (x, y) = create_data(&sorted_records, 72, 3, 168f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(30),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 40,
        m: Some(5),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
    // Calculate test error
    println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf).sqrt());
     */

    // 7 day data predict after 24 hours
    /*
    let (x, y) = create_data(&sorted_records, 168, 12, 24f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.00001f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    println!(
        "MSE: {}",
        mean_squared_error(&y_test, &y_hat_svm).sqrt()
    );
     */
    // Random Forest
/*
    let (x, y) = create_data(&sorted_records, 168, 1, 24f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(30),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 50,
        m: Some(9),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
    // Calculate test error
    println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf).sqrt());
*/
    // 7 day data predict after 72 hour
    /*
    let (x, y) = create_data(&sorted_records, 168, 12, 72f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.00001f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    println!(
        "MSE: {}",
        mean_squared_error(&y_test, &y_hat_svm).sqrt()
    );
     */
    // Random Forest
    /*
    let (x, y) = create_data(&sorted_records, 168, 2, 72f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(30),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 40,
        m: Some(6),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
    // Calculate test error
    println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf).sqrt());
     */
    // 7 day data predict after 7 days
/*
    let (x, y) = create_data(&sorted_records, 168, 12, 168f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.0005f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    println!(
        "MSE: {}",
        mean_squared_error(&y_test, &y_hat_svm).sqrt()
    );
 */
    // Random Forest
/*
    let (x, y) = create_data(&sorted_records, 168, 2, 168f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(30),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 50,
        m: Some(5),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
    // Calculate test error
    println!("MSE: {}", mean_squared_error(&y_test, &y_hat_rf).sqrt());
*/







    //precipitation prediction

    // 24 hour data, predict after 24 hours
/*
    let (x, y) = create_data_1(&sorted_records, 24, 5, 24f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 10f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.001f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_svm.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_svm[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_svm {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
 */
    // Random Forest
/*
    let (x, y) = create_data_1(&sorted_records, 24, 1, 24f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(10),
        min_samples_leaf: 1,
        min_samples_split: 1,
        n_trees: 10,
        m: Some(2),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();

        // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_rf.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_rf[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_rf {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/
    // 24 hour data predict after 72 hour
/*
    let (x, y) = create_data_1(&sorted_records, 24, 5, 72f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.01f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_svm.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_svm[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_svm {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/
    // Random Forest
/*
    let (x, y) = create_data_1(&sorted_records, 24, 1, 72f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(20),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 10,
        m: Some(1),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
        // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_rf.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_rf[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_rf {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/
    // 24 hour data predict after 7 days
/*
    let (x, y) = create_data_1(&sorted_records, 24, 5, 168f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.01f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_svm.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_svm[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_svm {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/
    // Random Forest
/*
    let (x, y) = create_data_1(&sorted_records, 24, 2, 168f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(20),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 20,
        m: Some(2),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
            // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_rf.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_rf[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_rf {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));

*/
    // 72 hour data predict after 24 hours
/*
    let (x, y) = create_data_1(&sorted_records, 72, 12, 24f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.00005f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_svm.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_svm[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_svm {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/
    // Random Forest
/*
    let (x, y) = create_data_1(&sorted_records, 72, 2, 24f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(20),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 20,
        m: Some(1),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
        // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_rf.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_rf[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_rf {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/


    // 72 hour data predict after 72 hour
/*
    let (x, y) = create_data_1(&sorted_records, 72, 12, 72f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.01f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_svm.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_svm[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_svm {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/
    // Random Forest
/*
    let (x, y) = create_data_1(&sorted_records, 72, 2, 72f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(30),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 20,
        m: Some(1),
        keep_samples: false,
        seed: 0,
    };

    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
    // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_rf.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_rf[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_rf {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/

    // 72 hour data predict after 7 days
/*
    let (x, y) = create_data_1(&sorted_records, 72, 12, 168f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.005f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_svm.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_svm[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_svm {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/
    // Random Forest
/*
    let (x, y) = create_data_1(&sorted_records, 72, 1, 168f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(10),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 10,
        m: Some(3),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
    // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_rf.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_rf[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_rf {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/

    // 7 day data predict after 24 hours
/*
    let (x, y) = create_data_1(&sorted_records, 168, 12, 24f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.00005f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_svm.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_svm[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_svm {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/
    // Random Forest
    /*
    let (x, y) = create_data_1(&sorted_records, 168, 4, 24f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(10),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 10,
        m: Some(1),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
    // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_rf.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_rf[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_rf {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/

    // 7 day data predict after 72 hour
/*
    let (x, y) = create_data_1(&sorted_records, 168, 12, 72f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.00005f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_svm.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_svm[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_svm {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/
    // Random Forest
/*
    let (x, y) = create_data_1(&sorted_records, 168, 2, 72f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(30),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 20,
        m: Some(1),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
    // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_rf.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_rf[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_rf {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/
    // 7 day data predict after 7 days
    /*
    let (x, y) = create_data_1(&sorted_records, 168, 12, 168f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = &SVRParameters{
        eps: 0.1,
        c: 100f32,
        tol: 1e-3,
        kernel: Some(Box::from(Kernels::rbf().with_gamma(0.001f64))),
    };
    let svm = SVR::fit(&x_train, &y_train, parameters)
        .unwrap();

    let y_hat_svm = svm.predict(&x_test).unwrap();
    // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_svm.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_svm[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_svm {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));
*/

    // Random Forest

    let (x, y) = create_data_1(&sorted_records, 168, 1, 168f32);
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, Some(1000));
    let parameters = RandomForestRegressorParameters{
        max_depth: Some(20),
        min_samples_leaf: 1,
        min_samples_split: 2,
        n_trees: 20,
        m: Some(3),
        keep_samples: false,
        seed: 0,
    };
    let rf = RandomForestRegressor::fit(&x_train, &y_train, parameters).unwrap();
    let y_hat_rf = rf.predict(&x_test).unwrap();
    // Calculate test error
    let mut labels: Vec<f32> = Vec::new();
    let mut true_labels: Vec<f32> = Vec::new();
    for i in 0..y_hat_rf.len() {
        if &y_test[i] != &0f32 {
            labels.push(y_hat_rf[i].clone());
            true_labels.push(y_test[i].clone());
        }
    }
    println!("MSE: {}", mean_squared_error(&true_labels, &labels).sqrt());
    let mut precipitation = 0;
    for label in &y_test {
        if label == &0f32 {
            precipitation += 1;
        }
    }
    let precipitation: f32 = precipitation as f32 / y_test.len() as f32;
    println!("{}", precipitation);

    let mut labels: Vec<f32> = Vec::new();
    for label in &y_hat_rf {
        if label == &0f32 {
            labels.push(0f32);
        } else {
            labels.push(1f32);
        }
    }

    let mut true_labels: Vec<f32> = Vec::new();

    for label in &y_test{
        if label == &0f32 {
            true_labels.push(0f32);
        } else {
            true_labels.push(1f32);
        }
    }

    println!("{}", precision(&true_labels, &labels));

}