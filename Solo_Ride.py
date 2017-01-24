import matplotlib.pyplot as plt
import pandas 
import statsmodels.formula.api as smf

# Woot!Woot! Gotta plot working 
# Important stuff , math.info() like the str of r, read_csv doesnt really work well
# type(math) returns the type of data, del is to delete
math = pandas.read_fwf('student_mat.txt')
#plot_1=pandas.DataFrame(math,columns=['G1','G2','G3'])
#math.plot.line('Fjob','traveltime')
#plot_1.plot.area()
#plt.show()

#Getting some models in: First Linear Regression
Y=math['G1']+math['G2']+math['G3']
math.append(Y,ignore_index=True)
lin1=smf.ols(formula='(Y/10)~age+Medu+Fedu+traveltime+studytime+health+absences',data=math).fit()
print lin1.params
print lin1.summary()
#Turn down for what!!!!!!*Bass Drop* 

#Next up is Logistic Regression, lets see how I can push this 
