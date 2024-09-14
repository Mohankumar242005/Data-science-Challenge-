# DATA TYPES

x=int(input("Enter the x value"))
y=int(input("Enter the y value"))
z=int(input("Enter the z value"))
string1="week1"
string2="Basics of python"
list=[1,2,3,4,5]
tuple=("hi",1,59,"connections")
dict={"Name": "Mohankumar","Age":20,"Gender":"Male"}

#ARITHMETIC OPERATOR
print("ARITHMETIC OPERATOR")
m=x+y*z
print(m)

cf=(((x+y)/2)+z)
print("cut off:",cf)

mk=(x*y-z)%2
mk1=mk**3
print(mk)
print(mk**5)

#COMPARISON OPERATOR
print("COMPARISON OPERATOR")
if(x==y):
    print("X equal to Y")
else:
    print("X  is not equal to Y")
if(x<=mk):
    print("X is lesser than or equal to mk")
else:
    print("X is greater than or equal to mk")
if(z!=mk1):
    print("Z is not equal to mk1")
else:
    print("Z is equal to mk")
    
#ASSIGNMENT OPERATOR
print("ASSIGNMENT OPERATOR")
a=x
a+=y
mk*=mk1
print(a)
print(mk)

#BITWISE OPERATOR
print("BITWISE OPERATOR")
print(x&y)
print(mk|mk1)
print(mk^mk1)
print(~mk)

#SHIFT OPERATORS
print("SHIFT OPERATORS")
mk>>1
mk<<1

#LOOPS AND CONDITIONS
print("LOOPS AND CONDITIONS")
if(a,y,z>50):
    print("Able to calculate")
    if(a,y,z>70):
        print("All the values are greater than 70")
elif(a,y,z>40):
    print("can calculate")
else:
    print("Cannot be calculate")
 
i=50
while(i<=55):
    print(i,end="")
    i=i+1

for i in range(10):
    print(i)

#FUNCTIONS
print("FUNCTIONS")

def swap(b,c):
    b,c=c,b
    print("value of b:",b,"value of c",c)
swap(56,78)

def area(x,y):
    print("Area of rectangle:")
    return (0.5*x*y)
area(89,60)
    