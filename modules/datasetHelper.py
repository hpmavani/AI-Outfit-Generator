import pandas as pd
import numpy as np

'''
Creates multi-indexed dataset out of Json-formatted data
'''
def createDataset(data_file: str): 
    iterables = []
    item_prices = []
    item_names = [] 
    item_catids = []
    item_setids = []

    import json
    with open(data_file) as file: 
        data = json.load(file)
        
    for index, d in enumerate(data): 
        setid = d["set_id"]
        items = [i for i in d["items"]]
        
        for i_index, i in enumerate(items): 
            iterables.append([index, i_index])
            item_names.append(i["name"])
            item_prices.append(i["price"])
            item_catids.append(i["categoryid"])
            item_index = i["index"]
            item_setids.append(f"{setid}_{item_index}")
    
    multindex = pd.DataFrame(iterables, columns=["outfit_index", "item_index"])
    index = pd.MultiIndex.from_frame(multindex)
    df = pd.DataFrame({"item_name": item_names, "item_price": item_prices, "item_catid": item_catids, 
                       "item_imageid": item_setids}, index=index)
    return preprocessDataset(df)
    

def preprocessDataset(df): 
    df.replace(["", " ", "...", "Polyvore", ''], np.nan, inplace=True)
    df["item_name"] = df["item_name"].str.lower()
    df["item_name"] = df["item_name"].str.replace("t shirt", "t-shirt")
    df["item_name"] = df["item_name"].str.replace(r"[^A-Za-z\s\-]", "", regex=True)
    df[df["item_name"] == ""]["item_name"] = np.nan
    df = df.dropna()
    df["tokenized_item"] = df["item_name"].apply(lambda x: x.split(" "))
    return df

# return combined_df and vocabulary list
def createVocabulary(df): 
    non_multi_df = df.reset_index()
    return non_multi_df, list(non_multi_df["item_name"]) 
     