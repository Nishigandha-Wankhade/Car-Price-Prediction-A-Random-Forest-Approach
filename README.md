# Feature-Importance-in-Vehicle-Price-Prediction-A-Random-Forest-Approach
This project aims to identify the most influential attributes in predicting advertised vehicle prices considering specific categories and characteristics of the vehicle. 

The data adapted from Huang et al. (2021), contains various features of the vehicle such as, who is the maker (company), vehicle model, model identification number, year advertised, advertised month, color, year made, vehicle body type, mileage, engine size in liters, transmission type, fuel type, vehicle price, along with the available seating and number of doors. 


Data Source Reference:
Huang, J., Chen, B., Luo, L., Yue, S., & Ounis, I. (2021). DVM-CAR: A large-scale automotive dataset for visual marketing research and applications. arXiv. https://doi.org/10.48550/arXiv.2109.00881



"""
=========================================================================================================
Project: "Feature Importance in Car Price Prediction: A Random Forest Approach"
Name: Nishigandha Wankhade
=========================================================================================================
"""
import pyreadr as pyr
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer # used with one hot encoding
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.inspection import permutation_importance
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")
carAd_file = pyr.read_r('C:\\Users\\VSC_Python\\car_ads_fp.RData')
carAd = carAd_file["carAd"]
# print(carAd) # will display all records from Dataset
carAd = pd.DataFrame(carAd) #convert data into dataframe using pandas
print(carAd) # will display all records from Dataframe
carAd.reset_index(drop = True, inplace = True)
              Maker Genmodel Genmodel_ID   Adv_ID  Adv_year  Adv_month  \
rownames                                                                 
1           Bentley   Arnage        10_1  10_1$$1    2018.0        4.0   
2           Bentley   Arnage        10_1  10_1$$2    2018.0        6.0   
3           Bentley   Arnage        10_1  10_1$$3    2017.0       11.0   
4           Bentley   Arnage        10_1  10_1$$4    2018.0        4.0   
5           Bentley   Arnage        10_1  10_1$$5    2017.0       11.0   
...             ...      ...         ...      ...       ...        ...   
268251    Westfield    Sport        97_1  97_1$$1    2018.0        5.0   
268252    Westfield    Sport        97_1  97_1$$2    2018.0        5.0   
268253        Zenos      E10        99_1  99_1$$1    2018.0        3.0   
268254        Zenos      E10        99_1  99_1$$2    2018.0        3.0   
268255        Zenos      E10        99_1  99_1$$3    2018.0        5.0   

           Color  Reg_year     Bodytype Runned_Miles Engin_size    Gearbox  \
rownames                                                                     
1         Silver    2000.0       Saloon        60000       6.8L  Automatic   
2           Grey    2002.0       Saloon        44000       6.8L  Automatic   
3           Blue    2002.0       Saloon        55000       6.8L  Automatic   
4          Green    2003.0       Saloon        14000       6.8L  Automatic   
5           Grey    2003.0       Saloon        61652       6.8L  Automatic   
...          ...       ...          ...          ...        ...        ...   
268251    Yellow    2006.0  Convertible         1800       2.2L     Manual   
268252    Yellow    2006.0  Convertible         2009        NaN     Manual   
268253       Red    2016.0  Convertible            6       2.0L     Manual   
268254     Green    2016.0  Convertible         1538       2.0L     Manual   
268255      Grey    2016.0  Convertible          500       2.3L     Manual   

         Fuel_type  Price  Seat_num  Door_num  
rownames                                       
1           Petrol  21500       5.0       4.0  
2           Petrol  28750       5.0       4.0  
3           Petrol  29999       5.0       4.0  
4           Petrol  34948       5.0       4.0  
5           Petrol  26555       5.0       4.0  
...            ...    ...       ...       ...  
268251      Petrol   8750       2.0       NaN  
268252         NaN   7995       NaN       NaN  
268253      Petrol  27950       2.0       NaN  
268254      Petrol  34950       2.0       NaN  
268255      Petrol  29995       2.0       NaN  

[268255 rows x 16 columns]
print(carAd.isnull().sum())    # displays the count for the number of null values in each column
carAd.info()
Maker               0
Genmodel            0
Genmodel_ID         0
Adv_ID              0
Adv_year            0
Adv_month           0
Color           21875
Reg_year            7
Bodytype          954
Runned_Miles        7
Engin_size       2064
Gearbox           167
Fuel_type         409
Price               0
Seat_num         6474
Door_num         4553
dtype: int64
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 268255 entries, 0 to 268254
Data columns (total 16 columns):
 #   Column        Non-Null Count   Dtype  
---  ------        --------------   -----  
 0   Maker         268255 non-null  object 
 1   Genmodel      268255 non-null  object 
 2   Genmodel_ID   268255 non-null  object 
 3   Adv_ID        268255 non-null  object 
 4   Adv_year      268255 non-null  float64
 5   Adv_month     268255 non-null  float64
 6   Color         246380 non-null  object 
 7   Reg_year      268248 non-null  float64
 8   Bodytype      267301 non-null  object 
 9   Runned_Miles  268248 non-null  object 
 10  Engin_size    266191 non-null  object 
 11  Gearbox       268088 non-null  object 
 12  Fuel_type     267846 non-null  object 
 13  Price         268255 non-null  object 
 14  Seat_num      261781 non-null  float64
 15  Door_num      263702 non-null  float64
dtypes: float64(5), object(11)
memory usage: 32.7+ MB
# Define makers and models of interest
makers_of_interest = ["Mitsubishi", "Audi", "Mazda", "Volvo", "BMW", "Ferrari"]
models_of_interest = ["L200", "Q3", "CX-5", "XC90", "6 Series Gran Coupe", "Enzo", "Laferrari", "488"]
body_type_of_interest = ["SUV", "Pickup", "Coupe", "Convertible", "Hatchback"]

# Filter data for specific makers and models
carAd_makers_r1 = carAd[(carAd["Maker"].isin(makers_of_interest))]     # 40560
carAd_models_r2 = carAd_makers_r1[(carAd_makers_r1["Genmodel"].isin(models_of_interest))]    # 4801
print(f"\n\t\t ============== r1 Makers:======================\n {carAd_makers_r1}")
print(f"\n\t\t =============== r2 Models: =====================\n {carAd_models_r2}")


carAd_body_type_r3 = carAd_models_r2[(carAd_models_r2["Bodytype"].isin(body_type_of_interest))]       # 4736
# Display the filtered data as per bodytype = Pickup
print(f"\n\t\t ===================r3 Body Types: =====================\n {carAd_body_type_r3}")


# Creating a List of all colors from Filtered makers data
print(carAd_makers_r1["Color"].value_counts(dropna = False) ) #inspect
# Top 6 colors from filterd model data
cols_list = carAd_makers_r1["Color"].value_counts().index.tolist()[:6]
print(f"\n\t\t =================== Color LIST: ================= \n {cols_list}")  #inspect

# Creating a List for two types of fuels i.e. diesel and petrol (unleaded)
print(carAd_makers_r1["Fuel_type"].value_counts(dropna = False) ) #inspect
# captures top 2 fuel types
fuel_type_list = carAd_makers_r1["Fuel_type"].value_counts().index.tolist()[:2]
print(f"\n\t\t =================== Fuel Type LIST: ====================== \n {fuel_type_list}") #inspect


carAd_colors_r4 = carAd_body_type_r3[carAd_body_type_r3["Color"].isin(cols_list)]
# Display the filtered data as per top 6 Colors
print(f"\n\t\t =============  r4 Colors ===\n {carAd_colors_r4["Color"].value_counts(dropna = False)}")  #inspect

carAd_fuel_r5 = carAd_colors_r4[carAd_colors_r4["Fuel_type"].isin(fuel_type_list)]
# Display the filtered data as per Fuel Types (Petrol or Diesel)
print(f"\n\t\t =================== r5 Fuel Types: =====================\n {carAd_fuel_r5}")
		 ============== r1 Makers:======================
           Maker Genmodel Genmodel_ID     Adv_ID  Adv_year  Adv_month   Color  \
14131   Ferrari      599       27_10   27_10$$1    2017.0        9.0    Grey   
14132   Ferrari      599       27_10   27_10$$2    2018.0        5.0     Red   
14133   Ferrari      599       27_10   27_10$$3    2018.0        4.0   Black   
14134   Ferrari      599       27_10   27_10$$4    2017.0        6.0    Blue   
14135   Ferrari      599       27_10   27_10$$5    2018.0        4.0     Red   
...         ...      ...         ...        ...       ...        ...     ...   
268245    Volvo      V50        96_9  96_9$$522    2018.0        7.0    Grey   
268246    Volvo      V50        96_9  96_9$$523    2018.0        8.0    Blue   
268247    Volvo      V50        96_9  96_9$$524    2018.0        5.0  Silver   
268248    Volvo      V50        96_9  96_9$$525    2018.0        5.0  Silver   
268249    Volvo      V50        96_9  96_9$$526    2018.0        1.0    Blue   

        Reg_year Bodytype Runned_Miles Engin_size    Gearbox Fuel_type  \
14131     2009.0    Coupe        27800       6.0L  Automatic    Petrol   
14132     2007.0    Coupe        23592       6.0L  Automatic    Petrol   
14133     2007.0    Coupe        20000       6.0L  Automatic    Petrol   
14134     2007.0    Coupe        41000       6.0L  Automatic    Petrol   
14135     2007.0    Coupe        28800       6.0L  Automatic    Petrol   
...          ...      ...          ...        ...        ...       ...   
268245    2008.0   Estate       140000       2.0L     Manual    Diesel   
268246    2007.0   Estate       158000       2.4L  Automatic    Diesel   
268247    2009.0   Estate        94000       2.4L  Automatic    Diesel   
268248    2004.0   Estate       111000       2.4L  Automatic    Petrol   
268249    2009.0   Estate       107000       2.0L     Manual    Diesel   

         Price  Seat_num  Door_num  
14131   106750       2.0       2.0  
14132    89950       2.0       2.0  
14133   119995       2.0       2.0  
14134    89950       2.0       2.0  
14135    79500       2.0       2.0  
...        ...       ...       ...  
268245    3185       5.0       5.0  
268246    2990       5.0       5.0  
268247    4250       5.0       5.0  
268248    2895       5.0       5.0  
268249    3684       5.0       5.0  

[58363 rows x 16 columns]

		 =============== r2 Models: =====================
           Maker   Genmodel Genmodel_ID      Adv_ID  Adv_year  Adv_month  \
14275   Ferrari       Enzo       27_15    27_15$$1    2017.0       10.0   
14276   Ferrari       Enzo       27_15    27_15$$2    2018.0        3.0   
14460   Ferrari  Laferrari       27_21    27_21$$1    2018.0        3.0   
14461   Ferrari  Laferrari       27_21    27_21$$2    2017.0       10.0   
14462   Ferrari  Laferrari       27_21    27_21$$3    2018.0        4.0   
...         ...        ...         ...         ...       ...        ...   
264898    Volvo       XC90       96_18  96_18$$915    2021.0        6.0   
264899    Volvo       XC90       96_18  96_18$$916    2021.0        6.0   
264900    Volvo       XC90       96_18  96_18$$917    2021.0        6.0   
264901    Volvo       XC90       96_18  96_18$$918    2021.0        6.0   
264902    Volvo       XC90       96_18  96_18$$919    2020.0       11.0   

         Color  Reg_year Bodytype Runned_Miles Engin_size    Gearbox  \
14275      Red    2003.0    Coupe         4397       6.0L        NaN   
14276      Red    2004.0    Coupe         7620       6.0L        NaN   
14460      Red    2014.0    Coupe         1824       6.3L        NaN   
14461      Red    2013.0    Coupe         5800       6.3L  Automatic   
14462      Red    2014.0    Coupe         1502        NaN        NaN   
...        ...       ...      ...          ...        ...        ...   
264898  Silver    2019.0      SUV        16670       2.0L  Automatic   
264899   Black    2019.0      SUV        34231       2.0L  Automatic   
264900  Silver    2019.0      SUV         7651       2.0L  Automatic   
264901   Black    2019.0      SUV         8400       2.0L  Automatic   
264902   White    2019.0      SUV            0       2.0L  Automatic   

            Fuel_type    Price  Seat_num  Door_num  
14275          Petrol  2000000       2.0       NaN  
14276          Petrol  2200000       2.0       NaN  
14460             NaN  2599990       2.0       NaN  
14461             NaN  2195000       2.0       NaN  
14462          Petrol  2500000       2.0       NaN  
...               ...      ...       ...       ...  
264898         Petrol    39990       7.0       5.0  
264899         Diesel    36000       7.0       5.0  
264900         Diesel    53990       7.0       5.0  
264901         Diesel    46990       7.0       5.0  
264902  Petrol Hybrid    51999       7.0       5.0  

[5409 rows x 16 columns]

		 ===================r3 Body Types: =====================
           Maker   Genmodel Genmodel_ID      Adv_ID  Adv_year  Adv_month  \
14275   Ferrari       Enzo       27_15    27_15$$1    2017.0       10.0   
14276   Ferrari       Enzo       27_15    27_15$$2    2018.0        3.0   
14460   Ferrari  Laferrari       27_21    27_21$$1    2018.0        3.0   
14461   Ferrari  Laferrari       27_21    27_21$$2    2017.0       10.0   
14462   Ferrari  Laferrari       27_21    27_21$$3    2018.0        4.0   
...         ...        ...         ...         ...       ...        ...   
264898    Volvo       XC90       96_18  96_18$$915    2021.0        6.0   
264899    Volvo       XC90       96_18  96_18$$916    2021.0        6.0   
264900    Volvo       XC90       96_18  96_18$$917    2021.0        6.0   
264901    Volvo       XC90       96_18  96_18$$918    2021.0        6.0   
264902    Volvo       XC90       96_18  96_18$$919    2020.0       11.0   

         Color  Reg_year Bodytype Runned_Miles Engin_size    Gearbox  \
14275      Red    2003.0    Coupe         4397       6.0L        NaN   
14276      Red    2004.0    Coupe         7620       6.0L        NaN   
14460      Red    2014.0    Coupe         1824       6.3L        NaN   
14461      Red    2013.0    Coupe         5800       6.3L  Automatic   
14462      Red    2014.0    Coupe         1502        NaN        NaN   
...        ...       ...      ...          ...        ...        ...   
264898  Silver    2019.0      SUV        16670       2.0L  Automatic   
264899   Black    2019.0      SUV        34231       2.0L  Automatic   
264900  Silver    2019.0      SUV         7651       2.0L  Automatic   
264901   Black    2019.0      SUV         8400       2.0L  Automatic   
264902   White    2019.0      SUV            0       2.0L  Automatic   

            Fuel_type    Price  Seat_num  Door_num  
14275          Petrol  2000000       2.0       NaN  
14276          Petrol  2200000       2.0       NaN  
14460             NaN  2599990       2.0       NaN  
14461             NaN  2195000       2.0       NaN  
14462          Petrol  2500000       2.0       NaN  
...               ...      ...       ...       ...  
264898         Petrol    39990       7.0       5.0  
264899         Diesel    36000       7.0       5.0  
264900         Diesel    53990       7.0       5.0  
264901         Diesel    46990       7.0       5.0  
264902  Petrol Hybrid    51999       7.0       5.0  

[5343 rows x 16 columns]
Color
Black          12237
Grey            9336
Blue            8275
White           8048
Silver          7699
NaN             6243
Red             4197
Green            428
Orange           398
Brown            306
Bronze           284
Gold             236
Yellow           224
Multicolour      152
Purple           151
Beige             92
Maroon            44
Burgundy           4
Turquoise          3
Navy               2
Pink               2
Magenta            2
Name: count, dtype: int64

		 =================== Color LIST: ================= 
 ['Black', 'Grey', 'Blue', 'White', 'Silver', 'Red']
Fuel_type
Diesel                             35574
Petrol                             21511
Hybrid  Petrol/Electric Plug-in      646
Electric                             253
Petrol Plug-in Hybrid                196
NaN                                   47
Diesel Hybrid                         44
Hybrid  Diesel/Electric Plug-in       37
Hybrid  Petrol/Electric               32
Petrol Hybrid                         14
Hybrid  Diesel/Electric                4
Bi Fuel                                3
Petrol Ethanol                         2
Name: count, dtype: int64

		 =================== Fuel Type LIST: ====================== 
 ['Diesel', 'Petrol']

		 =============  r4 Colors ===
 Color
Black     953
White     702
Grey      675
Blue      645
Silver    586
Red       468
Name: count, dtype: int64

		 =================== r5 Fuel Types: =====================
           Maker   Genmodel Genmodel_ID      Adv_ID  Adv_year  Adv_month  \
14275   Ferrari       Enzo       27_15    27_15$$1    2017.0       10.0   
14276   Ferrari       Enzo       27_15    27_15$$2    2018.0        3.0   
14462   Ferrari  Laferrari       27_21    27_21$$3    2018.0        4.0   
14588   Ferrari        488        27_7     27_7$$1    2018.0        8.0   
14589   Ferrari        488        27_7     27_7$$2    2018.0        3.0   
...         ...        ...         ...         ...       ...        ...   
264896    Volvo       XC90       96_18  96_18$$913    2021.0        6.0   
264898    Volvo       XC90       96_18  96_18$$915    2021.0        6.0   
264899    Volvo       XC90       96_18  96_18$$916    2021.0        6.0   
264900    Volvo       XC90       96_18  96_18$$917    2021.0        6.0   
264901    Volvo       XC90       96_18  96_18$$918    2021.0        6.0   

         Color  Reg_year Bodytype Runned_Miles Engin_size    Gearbox  \
14275      Red    2003.0    Coupe         4397       6.0L        NaN   
14276      Red    2004.0    Coupe         7620       6.0L        NaN   
14462      Red    2014.0    Coupe         1502        NaN        NaN   
14588     Grey    2016.0    Coupe         3000       3.9L  Automatic   
14589      Red    2016.0    Coupe         3450       3.9L  Automatic   
...        ...       ...      ...          ...        ...        ...   
264896   Black    2019.0      SUV        11368       2.0L  Automatic   
264898  Silver    2019.0      SUV        16670       2.0L  Automatic   
264899   Black    2019.0      SUV        34231       2.0L  Automatic   
264900  Silver    2019.0      SUV         7651       2.0L  Automatic   
264901   Black    2019.0      SUV         8400       2.0L  Automatic   

       Fuel_type    Price  Seat_num  Door_num  
14275     Petrol  2000000       2.0       NaN  
14276     Petrol  2200000       2.0       NaN  
14462     Petrol  2500000       2.0       NaN  
14588     Petrol   229995       2.0       2.0  
14589     Petrol   214993       2.0       2.0  
...          ...      ...       ...       ...  
264896    Diesel    44995       7.0       5.0  
264898    Petrol    39990       7.0       5.0  
264899    Diesel    36000       7.0       5.0  
264900    Diesel    53990       7.0       5.0  
264901    Diesel    46990       7.0       5.0  

[3960 rows x 16 columns]
print(f"\n\n======== CHECKING NULL VALUES ================== \n{final_sample_data1.isnull().sum()}")    # displays the count for the number of null values in each column
final_sample_data1 = carAd_fuel_r5
#final_sample_data2 = carAd_fuel_r5

======== CHECKING NULL VALUES ================== 
Maker           0
Genmodel        0
Genmodel_ID     0
Adv_year        0
Adv_month       0
Color           0
Reg_year        0
Bodytype        0
Runned_Miles    0
Engin_size      0
Gearbox         0
Fuel_type       0
Price           0
Seat_num        0
dtype: int64
# Typecasting of 'advertised years' from "float64" to an "integer" type
final_sample_data1['Adv_year'] = pd.to_numeric(final_sample_data1['Adv_year'], downcast = "integer")
final_sample_data1['Adv_month'] = pd.to_numeric(final_sample_data1['Adv_month'], downcast = "integer")
final_sample_data1['Reg_year'] = pd.to_numeric(final_sample_data1['Reg_year'], downcast = "integer")

final_sample_data1["Runned_Miles"] = final_sample_data1["Runned_Miles"].astype(float)

# -------------------- Cleaning Engine_size -------------------------


# Ensure the column is treated as a string type before processing
final_sample_data1['Engin_size'] = final_sample_data1['Engin_size'].astype(str)

# Remove the 'L' from the Engine_Size column
final_sample_data1['Engin_size'] = final_sample_data1['Engin_size'].str.replace('L', '')


# Convert the cleaned values to float if necessary
final_sample_data1['Engin_size'] = final_sample_data1['Engin_size'].astype(float)
mean_engine_size = final_sample_data1['Engin_size'].mean()
final_sample_data1['Engin_size'].fillna(mean_engine_size, inplace=True)


# changing 'Price' from "object" to "numeric" type and HANDLING any UNKNOWS values if any.
#================== REPLACING NULL / EMPTY VALUES OF "Price" with it's mean() ======================
# Step 1: Identify rows with valid numerical prices
# Convert Price column to numeric, forcing invalid entries to NaN
final_sample_data1["Price"] = pd.to_numeric(final_sample_data1["Price"], errors='coerce')

# Step 2: Calculate the mean of valid prices
mean_price = final_sample_data1['Price'].mean()

# Step 3: Replace NaN (invalid prices) with the mean
final_sample_data1['Price'].fillna(mean_price, inplace=True)


# Handle non-numerical values in the 'Price' column by replacing them with the mean
final_sample_data1['Seat_num'] = pd.to_numeric(final_sample_data1['Seat_num'], errors='coerce')
final_sample_data1['Seat_num'].fillna(final_sample_data1['Seat_num'].mean(), inplace=True)

# Handle non-numerical values in the 'Price' column by replacing them with the mean
final_sample_data1['Door_num'] = pd.to_numeric(final_sample_data1['Door_num'], errors='coerce')
final_sample_data1['Door_num'].fillna(final_sample_data1['Door_num'].min(), inplace=True)


# Replace NaN or empty values in the "Gearbox" column with a default value
final_sample_data1["Gearbox"] = final_sample_data1["Gearbox"].fillna("Unknown")

# If you want to replace empty strings as well
final_sample_data1["Gearbox"] = final_sample_data1["Gearbox"].replace('', "Unknown")

final_sample_data1["Gearbox"] = final_sample_data1["Gearbox"].astype(object)


# =================== DROP the column "Adv_ID"=======================
columns_to_drop = ["Adv_ID", "Door_num"]
final_sample_data1 = final_sample_data1.drop(columns = columns_to_drop)
print(final_sample_data1.info())
print(final_sample_data1.head())
<class 'pandas.core.frame.DataFrame'>
Index: 3960 entries, 14275 to 264901
Data columns (total 14 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Maker         3960 non-null   object 
 1   Genmodel      3960 non-null   object 
 2   Genmodel_ID   3960 non-null   object 
 3   Adv_year      3960 non-null   int16  
 4   Adv_month     3960 non-null   int8   
 5   Color         3960 non-null   object 
 6   Reg_year      3960 non-null   int16  
 7   Bodytype      3960 non-null   object 
 8   Runned_Miles  3960 non-null   float64
 9   Engin_size    3960 non-null   float64
 10  Gearbox       3960 non-null   object 
 11  Fuel_type     3960 non-null   object 
 12  Price         3960 non-null   float64
 13  Seat_num      3960 non-null   float64
dtypes: float64(4), int16(2), int8(1), object(7)
memory usage: 390.6+ KB
None
         Maker   Genmodel Genmodel_ID  Adv_year  Adv_month Color  Reg_year  \
14275  Ferrari       Enzo       27_15      2017         10   Red      2003   
14276  Ferrari       Enzo       27_15      2018          3   Red      2004   
14462  Ferrari  Laferrari       27_21      2018          4   Red      2014   
14588  Ferrari        488        27_7      2018          8  Grey      2016   
14589  Ferrari        488        27_7      2018          3   Red      2016   

      Bodytype  Runned_Miles  Engin_size    Gearbox Fuel_type      Price  \
14275    Coupe        4397.0    6.000000    Unknown    Petrol  2000000.0   
14276    Coupe        7620.0    6.000000    Unknown    Petrol  2200000.0   
14462    Coupe        1502.0    2.328123    Unknown    Petrol  2500000.0   
14588    Coupe        3000.0    3.900000  Automatic    Petrol   229995.0   
14589    Coupe        3450.0    3.900000  Automatic    Petrol   214993.0   

       Seat_num  
14275       2.0  
14276       2.0  
14462       2.0  
14588       2.0  
14589       2.0  
#print(final_sample_data1.info())
final_sample_data1.isnull().sum()
Maker           0
Genmodel        0
Genmodel_ID     0
Adv_year        0
Adv_month       0
Color           0
Reg_year        0
Bodytype        0
Runned_Miles    0
Engin_size      0
Gearbox         0
Fuel_type       0
Price           0
Seat_num        0
dtype: int64
# ============ Identifying fields requiring ENCODING ===================
objs = final_sample_data1.select_dtypes(include = np.object_).columns.tolist()
#print(objs) #inspect

# Separate features and target variable
x = final_sample_data1.drop(columns=['Price'])
y = final_sample_data1['Price']

# Step 2: Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['number']).columns

print(numerical_columns)

print(categorical_columns)
Index(['Adv_year', 'Adv_month', 'Reg_year', 'Runned_Miles', 'Engin_size',
       'Seat_num'],
      dtype='object')
Index(['Maker', 'Genmodel', 'Genmodel_ID', 'Color', 'Bodytype', 'Gearbox',
       'Fuel_type'],
      dtype='object')
# Step 3: Create preprocessing pipelines
# For numerical features: standard scaling
numerical_transformer = StandardScaler()

# For categorical features: one-hot encoding
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine transformers in a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, ['Runned_Miles', 'Adv_year', 'Adv_month', 'Reg_year', 'Engin_size', 'Seat_num']),
        ('cat', categorical_transformer, ['Maker', 'Genmodel', 'Genmodel_ID','Color', 'Bodytype', 'Gearbox', 'Fuel_type'])
    ]
)
# Step 4: Create a pipeline
# Combine preprocessing and a Random Forest Regressor model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Step 5: Split the data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# Step 6: Train the model
model.fit(x_train, y_train)

print("1. R\N{SUPERSCRIPT TWO} of Training Data = ", model.score(x_train, y_train))  # calculates R^2 (Coefficient of Determination): measures variance 
trPr = model.predict(x_train)  # Predicts car prices for the training set
print("2. RMSE of Training Data= ", metrics.mean_squared_error(y_train, trPr, squared = False)) # compute RMSE 

print(f"3. Target Variable (Price) Mean: {final_sample_data1['Price'].mean()}")
print(f"4. Target Variable (Price) STD: {final_sample_data1['Price'].std()}")
1. R² of Training Data =  0.9443244750392347
2. RMSE of Training Data=  17058.174095157116
3. Target Variable (Price) mean : 28090.497990892043
4. Target Variable (Price) STD : 71180.73195125385
"""
1. R² of Training Data =  0.9443
 - This means that the model explains approximately 94.43% of the variance in the training dataset.
   A high R² indicates that the model fits the training data well, and captures most of the relationships
   between the features and the target variable (Price).


2. RMSE of Training Data=  17058.17
   - RMSE provides a practical measure of prediction error in the target variable (price).
   RMSE = 17058.17 suggests that, on average, the model's predictions deviate from the actual prices by this amount.

3. Target Variable (Price) Mean: 28090.498
    - The average price of vehicles in your dataset is 28,090.50 units. This represents the typical price across all vehicles.

4. Target Variable (Price) STD: 71180.732
    - The standard deviation of 71,180.73 indicates that vehicle prices are widely spread out around the mean, showing a lot of variation. 
    The high standard deviation relative to the mean suggests that there are many high-priced vehicles, which might be skewing the dataset.

"""
# Step 7: Make predictions
y_pred = model.predict(x_test)
# Step 8: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Model Evaluation Metrics for Test Data:")
print(f"1. Mean Absolute Error (MAE) for Test Data: {mae}\n")
print(f"2. Mean Squared Error (MSE) for Test Data: {mse}\n")
print(f"3. Root Mean Squared Error (RMSE) of Test Data: {rmse}\n")
print(f"4. R\N{SUPERSCRIPT TWO} of Test Data: {r2}\n")
Model Evaluation Metrics for Test Data:
1. Mean Absolute Error (MAE) for Test Data: 1863.14208795144

2. Mean Squared Error (MSE) for Test Data: 41141788.106254585

3. Root Mean Squared Error (RMSE) of Test Data: 6414.186472675595

4. R² of Test Data: 0.9912265612579674

"""
                                           
1. Mean Absolute Error (MAE) for Test Data:: 1863.14
    - It measures the average absolute errors between the predicted and actual values. 
    MAE = 1863.14 shows, on average, the model's predictions are off by about 1863 units in price. 
    This suggests relatively small errors, which implies the model performs well on the test data.

2. Mean Squared Error (MSE) for Test Data:: 41141788.12
    - It represents the average of the squared differences between predicted and actual values.
    The larger the values, the greater the presence of larger errors in some predictions. 
    
3. Root Mean Squared Error (RMSE) of Test Data: 6414.19
    - - RMSE provides a practical measure of prediction error in the target variable (price).
   RMSE =6414.19 suggests that, on average, the model's predictions are off by 6414 units in price. 
   Which is more substantial that the MSE.


4. R² of Test Data: 0.9912
    - It means that approximately 99.12% of the variance in the test data is explained by the model.
    This is an excellent score, reflects that the model performs very well in predicting prices on the test dataset, 
    with very high explanatory power.

MAE is relatively low, meaning the model's predictions are quite close to the actual values on average.

"""
# =============== VISUALIZING TESTING PERFORMANCE =====================
fig, ax = plt.subplots()
ax.scatter(y_pred, y_test, edgecolors = (0, 0, 1))
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--', lw = 3)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
plt.show(block = True)
No description has been provided for this image
"""
The above scatter plot shows the perfect linear relationship between the actual values and the predicted values of the vehicle price,
in the presence of a few exceptional outliers.
"""
# =============== EVALUATING IMPORTANCE FROM REGRESSION MODEL (IMPURITY) ===============
rf_imps = model.named_steps["regressor"].feature_importances_
print(rf_imps) #inspect
[1.03736269e-02 5.65993261e-04 7.12739829e-04 8.51262728e-03
 4.52536618e-02 6.42237317e-04 1.47318883e-05 2.63124527e-04
 1.07695572e-01 2.67471299e-05 4.27859432e-06 1.78442168e-03
 1.13535174e-01 2.94717360e-04 2.21635116e-05 2.74004600e-02
 4.70127219e-06 2.79240042e-02 1.94710666e-05 7.88406637e-04
 5.15245198e-02 2.65398110e-02 1.12202870e-01 2.38701347e-05
 5.23930365e-06 2.12806365e-05 3.55164917e-04 2.01351607e-03
 1.51197737e-04 7.03709610e-05 6.18222812e-05 9.76784694e-05
 7.49008047e-05 6.61774700e-05 1.43645254e-03 1.77784285e-03
 1.99392140e-08 4.17986777e-06 7.71406539e-06 1.98067335e-04
 4.14254122e-04 4.57055250e-01 2.62843289e-05 3.26551295e-05]
print(model.named_steps.keys())  # Check the steps in the pipeline
print(f"\n\n {model.named_steps["preprocessor"].named_transformers_}")    # Check transformer keys
dict_keys(['preprocessor', 'regressor'])


 {'num': StandardScaler(), 'cat': OneHotEncoder(handle_unknown='ignore')}
#print(model.named_steps["preprocessor"].named_transformers_["preprocessor"].categories_) # a list of lists of encoded names

one_hot_cat = model.named_steps["preprocessor"].named_transformers_["cat"].categories_  # a list of all column names of independent variables

# indeps has column names except 'Price':
indeps = final_sample_data1.iloc[:, final_sample_data1.columns != "Price"].columns.tolist()

# list to house the labels that were used in the model
features = []

#objs has categorical column names
for each in indeps:      # create label list for importance
    if each in objs:     # objs has categorical column names
        spot = objs.index(each)        # get position index
        # for each one hot lable, prefix column name and underscore to label
        one_hot_mess = [each + "_" + str(item) for item in one_hot_cat[spot]]  # str() because of seats and doors
        features.extend(one_hot_mess)  # extend a list with another iterable
    else:
        features.append(each)     # append a single item to the list
# Features contain all distinct categories from 4 cat variables and 2 numeric variables: total 16 columns
print(features)     # feature labels sent to model
['Maker_Audi', 'Maker_BMW', 'Maker_Ferrari', 'Maker_Mazda', 'Maker_Mitsubishi', 'Maker_Volvo', 'Genmodel_488', 'Genmodel_6 Series Gran Coupe', 'Genmodel_CX-5', 'Genmodel_Enzo', 'Genmodel_L200', 'Genmodel_Laferrari', 'Genmodel_Q3', 'Genmodel_XC90', 'Genmodel_ID_27_15', 'Genmodel_ID_27_21', 'Genmodel_ID_27_7', 'Genmodel_ID_57_7', 'Genmodel_ID_62_13', 'Genmodel_ID_7_19', 'Genmodel_ID_8_12', 'Genmodel_ID_96_18', 'Adv_year', 'Adv_month', 'Color_Black', 'Color_Blue', 'Color_Grey', 'Color_Red', 'Color_Silver', 'Color_White', 'Reg_year', 'Bodytype_Convertible', 'Bodytype_Coupe', 'Bodytype_Hatchback', 'Bodytype_Pickup', 'Bodytype_SUV', 'Runned_Miles', 'Engin_size', 'Gearbox_Automatic', 'Gearbox_Manual', 'Gearbox_Unknown', 'Fuel_type_Diesel', 'Fuel_type_Petrol', 'Seat_num']
one_hot_features = model.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out()   #inspect
print(f"\n One-hot encoded features: \n {one_hot_features}")
 One-hot encoded features: 
 ['Maker_Audi' 'Maker_BMW' 'Maker_Ferrari' 'Maker_Mazda' 'Maker_Mitsubishi'
 'Maker_Volvo' 'Genmodel_488' 'Genmodel_6 Series Gran Coupe'
 'Genmodel_CX-5' 'Genmodel_Enzo' 'Genmodel_L200' 'Genmodel_Laferrari'
 'Genmodel_Q3' 'Genmodel_XC90' 'Genmodel_ID_27_15' 'Genmodel_ID_27_21'
 'Genmodel_ID_27_7' 'Genmodel_ID_57_7' 'Genmodel_ID_62_13'
 'Genmodel_ID_7_19' 'Genmodel_ID_8_12' 'Genmodel_ID_96_18' 'Color_Black'
 'Color_Blue' 'Color_Grey' 'Color_Red' 'Color_Silver' 'Color_White'
 'Bodytype_Convertible' 'Bodytype_Coupe' 'Bodytype_Hatchback'
 'Bodytype_Pickup' 'Bodytype_SUV' 'Gearbox_Automatic' 'Gearbox_Manual'
 'Gearbox_Unknown' 'Fuel_type_Diesel' 'Fuel_type_Petrol']
# ============================= CREATING IMPORTANCE TABLE =================================
# Helps to find the most influential features among all 
influencers = pd.DataFrame({"Features": features,
                           "Importance": rf_imps}). sort_values(by = "Importance", 
                                                             ascending = False)
print(f"\n\n influencers :\n {influencers.head(10)}") 

 influencers :
              Features  Importance
41   Fuel_type_Diesel    0.457055
12        Genmodel_Q3    0.113535
22           Adv_year    0.112203
8       Genmodel_CX-5    0.107696
20   Genmodel_ID_8_12    0.051525
4    Maker_Mitsubishi    0.045254
17   Genmodel_ID_57_7    0.027924
15  Genmodel_ID_27_21    0.027400
21  Genmodel_ID_96_18    0.026540
0          Maker_Audi    0.010374
"""
The above results of Features and their Importance show us that, the most influential feature that will affect the vehicle price is "Fuel type". 
Followed with the "Model-Q3", "Advertised year", and "Model-CX-5" of the vehicle. 
The remaining features also show their impact but it's on a lower scale.
"""
# ====================== COLLECT PERMUTATIONS TO CROSS VERIFY Random Forest's PERFORMANCE / to VALIDATE FEATURES IMPORTANCE ===============================

#  to check weakness of Random Forest Algorithm
perm_imps = permutation_importance(model , x, y, n_repeats = 5, random_state = 1)

# create an array to use for sorting
perm_sorts = perm_imps.importances_mean.argsort()
print(perm_imps.importances[perm_sorts])      # inspect
[[7.01552462e-05 6.63473363e-05 7.94641215e-05 7.44369786e-05
  6.25898509e-05]
 [2.46317528e-04 2.35722118e-04 2.52990164e-04 2.60000804e-04
  2.48359755e-04]
 [3.63583370e-04 3.32741184e-04 5.31198827e-04 3.77541320e-04
  3.55569615e-04]
 [1.08690071e-04 1.41891688e-04 4.87525988e-05 2.97108542e-04
  1.46301990e-03]
 [5.51362124e-04 4.99156930e-04 4.73327132e-04 5.60344749e-04
  6.53522886e-04]
 [2.88610360e-03 3.01817384e-03 2.69717900e-03 2.80193018e-03
  2.83841877e-03]
 [7.50506319e-03 7.91111521e-03 7.65410438e-03 7.98078873e-03
  7.55783287e-03]
 [9.02779495e-03 9.06044044e-03 1.03238832e-02 9.50732149e-03
  8.87155700e-03]
 [1.34253189e-02 1.32051688e-02 1.32009166e-02 1.32947473e-02
  1.31571761e-02]
 [4.99249650e-02 4.99167750e-02 5.12294235e-02 4.79442214e-02
  4.99162939e-02]
 [6.35540674e-02 6.40426812e-02 6.54362591e-02 6.26170116e-02
  6.38716155e-02]
 [8.72960830e-02 8.75831143e-02 8.94273845e-02 8.60541362e-02
  8.73899708e-02]
 [7.52483365e-01 7.58018456e-01 7.62578618e-01 7.63580489e-01
  7.64802773e-01]]
# create an offset index for plotting
indices = np.arange(0, len(rf_imps)) + .5

# indeps has column names except 'Price'
# get features order for permutation
perms_df = pd.DataFrame({"Features": indeps, # indeps has column names except 'Price':
                        "Permutation": perm_imps.importances_mean}).sort_values(by = "Permutation",
                                                                               ascending = False)
                                              
perms_df
Features	Permutation
10	Gearbox	0.760293
2	Genmodel_ID	0.087550
1	Genmodel	0.063904
0	Maker	0.049786
9	Engin_size	0.013257
6	Reg_year	0.009358
8	Runned_Miles	0.007722
7	Bodytype	0.002848
5	Color	0.000548
4	Adv_month	0.000412
3	Adv_year	0.000392
12	Seat_num	0.000249
11	Fuel_type	0.000071
"""
 THE PERMUTATION IMPORTANCE METHOD EVALUATES FEATURE IMPORTANCE AFTER THE MODEL IS TRAINED. 
 FOR EACH FEATURE, IT RANDOMLY SHUFFLES THE VALUES AND MEASURES HOW THE SHUFFLING AFFECTS THE MODEL'S PERFORMANCE.
 IF A FEATURE IS IMPORTANT, SHUFFLING IT WILL CAUSE A SIGNIFICANT DROP IN PERFORMANCE.
 
======================   Top Influential Features after shuffling:    ============================

- Gearbox has the highest permutation importance (0.760293), meaning it is the most influential feature in predicting the target variable (Price). 
  Its removal would cause the most significant drop in model performance.

- Genmodel_ID (0.087550) and Genmodel (0.063904) also show moderate influence, suggesting the vehicle's specific model plays an important role in price 
  prediction.


========================  Moderately Influential Features: ================================

- Maker (0.049786) and Engine size (0.013257) still contribute to the prediction but with less impact compared to the top-ranked features.


=============================   Least Influential Features:  =======================================

- Features like Seat_num (0.000249), Fuel_type (0.000071), and Bodytype (0.002848) have near-zero permutation importance. 
  This suggests these features contribute minimally or are possibly redundant for the model.


===========================    Temporal Features: ===================================

- Reg_year (0.009358) has more importance than Adv_month (0.000412) or Adv_year (0.000392), indicating registration year is more relevant to the price 
  than when the advertisement was created.

IF WE CONSIDER THE LAST STATEMENT FROM THE PERMUTATION IMPORTANCE DESCRIPTION, THEN "FULE_TYPE" IS THE MOST IMPORTANT FEATURE IN THIS SCENARIO AS ITS
PERFORMANCE IS SIGNIFICANTLY CHANGED AS COMPARED TO THE IMPORTANCE FEATURES EXTRACTED WITH RANDOM FOREST REGRESSOR.
"""
# ================ IMPORTANCE VISUALIZATION =======================
# Visualizing differences between importance measures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 8))

# horizontal bar graph of original importance values
ax1.barh(indices, rf_imps[np.argsort(rf_imps)], height = .7)
ax1.set_yticks(indices)
# ax1.set_yticklabels(influencers["Features"].tolist().reverse())    # .reverse():- modifies the list in place and returns 'None', which caused the error
ax1.set_yticklabels(influencers["Features"].tolist()[::-1])    # or use this:-  ax1.set_yticklabels(list(reversed(influencers["Features"].tolist()))) 
                                                             # [::-1] or reversed(): Returns a reversed copy of the list, leaving the original intact

ax1.set_ylim(0, len(rf_imps))

# box plot showing permutation importance across n_repeats
ax2.boxplot(perm_imps.importances[perm_sorts].T,
           vert = False, labels = perms_df["Features"].tolist(),
           patch_artist = True,
           boxprops = {'color': "black", 'edgecolor': "black"})

fig.tight_layout()
plt.show(block = True) # displaying the graph
No description has been provided for this image
"""
- The Bar Plot gives a direct understanding of which features are most important in predicting the price, highlighting Fuel_type and Genmodel as 
key contributors.

- The Box Plot helps to understand the distribution of these features relative to the target variable, showing how some features like 
Fuel_type or Run_Miles exhibit clear price differences, while others like Bodytype and Color have less distinct effects.

- Together, the plots suggest that the model relies heavily on Fuel_type, Genmodel, and Engine size in determining vehicle prices, 
with other features like Bodytype and Color contributing less. This insight can help in marketing strategies, where emphasizing 
certain models, fuel types, or engine specifications can be used to target specific price ranges.

"""
# ================ CHECK Correlation Matrix ====================
numeric_data = final_sample_data1.select_dtypes(include=[np.number])  # selecting all numerical datatypes

# One-hot encoding for categorical columns
encoded_data = pd.get_dummies(final_sample_data1.select_dtypes(include=["object"]))
numeric_data_with_encoded = pd.concat([numeric_data, encoded_data], axis=1)

# Adjust figure size to avoid overlapping
plt.figure(figsize=(12, 10))  # Change width and height as needed

# ================ CHECK Correlation Matrix ====================
numeric_data = final_sample_data1.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
No description has been provided for this image
# Adjust figure size to avoid overlapping
plt.figure(figsize=(28, 18))  # Change width and height as needed


# Compute correlation matrix for all numeric and encoded columns
correlation_matrix = numeric_data_with_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 10})
plt.title("Correlation Matrix (Including All Features)", fontsize=12)
plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels and adjust font size
plt.yticks(fontsize=10)  # Adjust y-axis font size
plt.tight_layout()  # Ensure everything fits nicely
plt.show()
No description has been provided for this image
"""
you can compare Random Forest's built-in feature importance with permutation importance, but it's important to understand the differences between 
the two methods. Here's an explanation and approach:

1. Random Forest Feature Importance

How it works:
Random Forest uses Mean Decrease in Impurity (MDI) to calculate feature importance.
It measures the total reduction in impurity (e.g., Gini Impurity or Mean Squared Error) contributed by each feature across all trees in the forest.
Built-in importance values are biased toward features with more categories or higher variance.

Pros:
Fast to compute as it’s part of the training process.
Provides a quick ranking of features.

Cons:
Can be biased if the features differ in scale or type.
May not represent the actual effect of the feature when used in real-world data.


2. Permutation Importance

How it works:
Permutation Importance evaluates feature importance after the model is trained.
For each feature, it randomly shuffles the values and measures how the shuffling affects the model's performance (e.g., R² or RMSE).
If a feature is important, shuffling it will cause a significant drop in performance.

Pros:
Reflects the real-world impact of the feature on model predictions.
Less biased compared to MDI.

Cons:
Computationally expensive because it requires multiple evaluations of the model.
Sensitive to data leakage or correlated features.

"""
