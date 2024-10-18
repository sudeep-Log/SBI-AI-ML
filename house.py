import joblib, pandas as pd
def predict_price(df):
    house_data=df.copy()
    model = joblib.load("housemodel.h5")
    model_columns= model.feature_names_in_
    
    house_data=pd.get_dummies(data=house_data,columns=['Brick','Neighborhood'],
                             dtype=int,drop_first=True)
    
    missing_columns = list(set(model_columns)-set(house_data.columns))
    for i in missing_columns:
        house_data[i]=0
    data=house_data[model_columns]
    pred = model.predict(data)
    return pred