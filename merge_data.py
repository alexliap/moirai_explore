import pandas as pd


def load_data(data_path: str, country_code: str) -> pd.DataFrame:
    data = pd.read_csv(data_path, delimiter='\t')
    if len(data.columns) == 1:
        data = pd.read_csv(data_path, delimiter=',')
        
    if len(data.columns) == 1:
        raise RuntimeError("Find the correct delimiter.")
    
    data = data.query(f"CountryCode == '{country_code}'").reset_index()
    data = data.drop(["CreateDate", "UpdateDate", "CountryCode", "index", 
                      "MeasureItem", "DateShort", "TimeFrom", "TimeTo",
                      "Cov_ratio", "Value_ScaleTo100"], axis=1)
    data['DateUTC'] = pd.to_datetime(data['DateUTC'], dayfirst=True)
    
    return data


if __name__ == "__main__":
    # get data for Greece
    gr_data_2023 = load_data('data/load_values_23.csv', country_code='GR')
    gr_data_2023.iloc[2018, 0] = pd.Timestamp(2023, 3, 26, 2)

    gr_data_2024 = load_data('data/load_values_24.csv', country_code='GR')
    gr_data_2024.loc[len(gr_data_2024.index)] = [pd.Timestamp(2024, 3, 31, 2), gr_data_2024['Value'].mean()]
    gr_data_2024.loc[len(gr_data_2024.index)] = [pd.Timestamp(2024, 3, 31, 3), gr_data_2024['Value'].mean()]
    gr_data_2024 = gr_data_2024.sort_values('DateUTC')
    
    pd.concat([gr_data_2023, gr_data_2024]).set_index('DateUTC').to_csv("data/gr_load_data_23_24.csv")
    
    # get data for Italy
    it_data_2023 = load_data('data/load_values_23.csv', country_code='IT')
    it_data_2023.iloc[2018, 0] = pd.Timestamp(2023, 3, 26, 2)

    it_data_2024 = load_data('data/load_values_24.csv', country_code='IT')
    it_data_2024 = it_data_2024.sort_values('DateUTC')
    it_data_2024.iloc[2162, 0] = pd.Timestamp(2024, 3, 31, 2)

    pd.concat([it_data_2023, it_data_2024]).set_index('DateUTC').to_csv("data/it_load_data_23_24.csv")
    