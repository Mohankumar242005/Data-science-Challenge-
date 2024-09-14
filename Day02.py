import numpy as np
import pandas as pd
#one dimensional Array
print("NUMPY \n")
oneDarr=np.array([45,78,90,40])
print("One dimensional Array:\n",oneDarr)

#Two dimentional Array

twoDarr=np.array([[45,78,90,40],[78,34,45,67]])
print("Two dimensional Array:\n",twoDarr)

#Three dimentional Array

threeDarr=np.array([[[45,78,90,40],[78,34,45,67]],[[45,78,90,40],[78,34,45,67]]])
print("Three dimensional Array:\n",threeDarr)

print("Data of One Dimensional Array is",oneDarr.data)
print("Datatype of Two Dimensional Array is",twoDarr.dtype)
print("shape of Two Dimensional Array is",twoDarr.shape)
print("strides of Three Dimensional Array is",threeDarr.strides)

#Built-in NumPy functions

print("Built-in NumPy functions")

ones=np.ones((3,4))
print("3*4 Matrics with valu 1:\n",ones)

print("\nRandom Array:/n")
np.random.random((4,3))

#Inspecting Numpy Arrays

print("Inspecting Numpy Arrays")
print("Dimensions of threeDarr",threeDarr.ndim)
print("Elements in twoDarr",twoDarr)
print("Flags of oneDarr",oneDarr.flags)
print("ItemSize of twoDarr",twoDarr.itemsize)
print("Bytes of threeDarr",threeDarr.nbytes)

#Mathematical operations in Numpy

print("Mathematical operations in Numpy")

#Addition

oneDarr1=np.array([34,69,46,87])
add=np.add(oneDarr,oneDarr1)
print("Addition of oneDarr and oneDarr1  is",add)

sub=np.subtract(oneDarr,oneDarr1)
print("Multiplication of oneDarr and oneDarr1  is",sub)

multiply=np.multiply(oneDarr,oneDarr1)
print("Multiplication of oneDarr and oneDarr1  is",multiply)

divide=np.divide(oneDarr,oneDarr1)
print("Division of oneDarr and oneDarr1  is",divide)

#subset or Slicing

print("subset or Slicing")

print("Sliced set of oneDarr1 is",oneDarr1[0:2])
print("2Darray slicing:",twoDarr[0:2,1])
print("values greater than 20 in twoDarr",twoDarr>20)

print("PANDAS\n")

#Creating a Data Set
print("Creating a Data Set")
df=pd.DataFrame({
    "Name":["Mohankumar","Moorthy","Barath","Vinoth","Kowsik","Akash","Sakthikumar"],
    "Age":[20,20,19,19,20,19,19],
    "Dept":["AI","AI","AI","MECH","IT","EEE","AI"]
})
print(df)

#read a CSV file

print("Reading a CSV File")

df=pd.read_csv("iris.csv")
print("From Head:\n",df.head)
print("From Tail:\n",df.tail)
print("General information about the DataFrame\n",df.info())  
print("Summary statistics of numerical columns\n",df.describe())   
print(" List of column names\n",df.columns)  
print(df.shape) 
print(df[["Id","SepalLengthCm","PetalLengthCm"]])

print("Conditional sections")
print("Species with SepalLengthCm >3 & PetalWidthCm >4",df[(df["SepalLengthCm"] >3 ) & (df["PetalWidthCm"]>1)])
                                                          
print("\n Data Cleaning\n")
print("Check for missing values\n",df.isnull()) 
print("Check for missing values row wise\n",df.isnull().sum())
print("Remove rows with any missing values\n",df.dropna())
print("Returns a boolean Series indicating duplicates \n",df.duplicated()) 
print("Display data types of each column",df.dtypes)   
df.rename(columns={'PetalLengthCm': 'petallength'}, inplace=True)
print(df.head)
