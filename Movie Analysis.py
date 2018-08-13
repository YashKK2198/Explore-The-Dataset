
# coding: utf-8

# Introduction
# The primary goal of the project is to go through the general data analysis process — using basic data analysis technique with NumPy, pandas, and Matplotlib.
# I have been give data of movies from year 1966 to 2015 !
# It collects 5000+ movies basic move information and movie matrices, including user ratings, popularity and revenue data. These metrics can be seen as how successful these movies are. The movie basic information contained like cast, director, keywords, runtime, genres, etc.
# 

# Questions that are taken into account are :
# 
# 
# Question 1: Popularity Over Years
# 
# Question 2: Number of movie released year by year
# 
# Question 3: Show the variation of popularity in genre Comedy ?
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('tmdb-movies.csv')
df.head(1)


# In[3]:


# only column info
df.info()


# In[4]:


#some descriptive statistics for a data set

df.describe ()


# In[5]:


# Here I am filtering tohe random 3 movies whose budget is zero in the given csv file
df_budget_zero = df.query('budget == 0')
df_budget_zero.head(3)


# In[6]:



#Similarly done for revenue = 0 data

df_revenue_zero = df.query('revenue == 0')
df_revenue_zero.head(3)


# In[7]:


# zero values od budget
df_budget_0count =  df.groupby('budget').count()['id']
df_budget_0count.head(2)


# In[8]:


#zero values of revenue
df_revenue_0count =  df.groupby('revenue').count()['id']
df_revenue_0count.head(2)


# In[9]:


#delete duplicates
df.drop_duplicates(inplace=True)


# DATA WRANGLING 
# 
#  Categories
#  

# In[10]:


genre_details = list(map(str,(df['genres'])))
genre = []
for i in genre_details:
    split_genre = list(map(str, i.split('|')))
    for j in split_genre:
        if j not in genre:
            genre.append(j)
# printing list of seperated genres.

print(genre)


# In[11]:


# minimum range value
min_year = df['release_year'].min()
# maximum range value
max_year = df['release_year'].max()
print(min_year, max_year)


# NUMBER OF MOVIES RELEASED IN THEIR RESPECTIVE YEARS !

# In[12]:


genre_df = pd.DataFrame(index = genre, columns = range(min_year, max_year + 1))
# to fill not assigned values to zero
genre_df = genre_df.fillna(value = 0)


# In[13]:


# list of years of each movie
year = np.array(df['release_year'])
# index to access year value
z = 0
for i in genre_details:
    split_genre = list(map(str,i.split('|')))
    for j in split_genre:
        genre_df.loc[j, year[z]] = genre_df.loc[j, year[z]] + 1
    z+=1
genre_df


# 
# 
# 
# Oldest And The Latest Movie Release Years !

# QUESTION 1 > POPULARITY OVER THE YEARS !
# 
# Below is the popularity of the movies in latest five years

# In[14]:


df.head(2)


# In[15]:


p_median = df.groupby('release_year').median()['popularity']
p_median.tail()


# In[16]:


p_mean = df.groupby('release_year').mean()['popularity']
p_mean.tail()


# In[17]:


index_mean = p_mean.index
index_median = p_median.index


# In[18]:



sns.set_style('darkgrid')
x1, y1 = index_mean, p_mean
x2, y2 = index_median, p_median
plt.figure(figsize=(15, 8))
plt.scatter(x1, y1, color = 'b', label = 'mean')
plt.scatter(x2, y2, color = 'r', label = 'median')
plt.title('Popularity Over Years')
plt.xlabel('Year')
plt.ylabel('Popularity');
plt.legend(loc='upper left')


# Here we can see the mean distribution is greater than the median . The median distribution is falling year by year.
# 

# In[19]:


sns.set_style('ticks')
x1, y1 = index_mean, p_mean
x2, y2 = index_median, p_median
plt.figure(figsize=(9, 4))
plt.plot(x1, y1, color = 'b', label = 'mean')
plt.plot(x2, y2, color = 'r', label = 'median')
plt.title('Popularity Over Years')
plt.xlabel('Year')
plt.ylabel('Popularity');
plt.legend(loc='upper left')


# ANSWER --> As we go from left to right the popularity of mean is upward and the highest it is in the year 2015. 
#  Inversely the popularity of median has decreased from left to right.This is because there are other sources available where we can the reviews watch online movies etc.

# QUESTION 2--> Number of movie released year by year

# In[20]:


counter = df.groupby('release_year').count()['id']
counter.head()


# In[21]:


sns.set_style('white')
x = counter.index
y = counter
plt.figure(figsize=(10, 5))
plt.bar(x, y, color = 'g', label = 'mean')
plt.title('Number of Movie Released year by year')
plt.xlabel('Year')
plt.ylabel('Frequency');


# ANSWER 2 --> We can see from the above bar diagram that the frequency of number of movies releasing is increasing increasing
# exponentially. As we can see the number of movies released in 1960 is least , from there the graph goes on increasing.

# Properties are associated with movies that have high popularity?

# Question 3--> Budget level movie which are associated with movies that have high popularity?

# 
# 

# In[22]:



#dataframe with genre as index and years as columns to get a count of popularity


popularity_df = pd.DataFrame(index = genre, columns = range(min_year, max_year + 1))
# to fill not assigned values to zero
popularity_df = popularity_df.fillna(value = 0.0)


# In[23]:


#list of popularity levels of each movie

popularity = np.array(df['popularity'])  #numpy
# to check whether any popularity is zero.
print (len(popularity[popularity==0]))
# index to access year value
z = 0
for i in genre_details:
    split_genre = list(map(str,i.split('|')))
    for j in split_genre:
            popularity_df.loc[j, year[z]] = popularity_df.loc[j, year[z]] + popularity[z]
    z+=1


# In[24]:


#function to standardize the popularity of values in dataframe


def standardize(p):
    p_std = (p - p.mean()) / p.std(ddof = 0)
    return p_std


# In[25]:


popularity_std = standardize(popularity_df)


# In[26]:


#series to hold the popular genre for every year
sankya_genre = pd.Series(index = range(min_year, max_year + 1)) #pandas
sankya_genre.head()


# In[27]:


# Maximum standardized popularity value of genre
for i in range(min_year, max_year + 1):
    sankya_genre[i] = popularity_std[i].argmax()
sankya_genre


# In[28]:


# PLOTTING A GRAPH FOR GENRE COMEDY
plt.plot(popularity_std.loc['Comedy']) 
plt.xlabel('year')
plt.ylabel('popularity levels')
plt.title('Distribution of popularity for the genre Comedy over the years')
plt.axis([1960, 2015, 0, 3.5])
plt.show()


# ANSWER 3-->  The above graph shows the rise and fall of the genre COMEDY over the years. 
# There are steep rises and falls in the popularity levels of the genre. 
# 

# In[29]:


plt.plot(popularity_std.loc['Drama']) 
plt.xlabel('year')
plt.ylabel('popularity levels')
plt.title('Distribution of popularity for the genre Drama over the years')
plt.axis([1960, 2015, 0, 3.5])
plt.show()


# In[30]:


plt.plot(popularity_std.loc['Action']) 
plt.xlabel('year')
plt.ylabel('popularity levels')
plt.title('Distribution of popularity for the genre Action over the years')
plt.axis([1960, 2015, 0, 3.5])
plt.show()


# In[31]:


plt.plot(popularity_std.loc['Thriller']) 
plt.xlabel('year')
plt.ylabel('popularity levels')
plt.title('Distribution of popularity for the genre Thriller over the years')
plt.axis([1960, 2015, 0, 3.5])
plt.show()


# In[32]:


plt.plot(popularity_std.loc['Adventure']) 
plt.xlabel('year')
plt.ylabel('popularity levels')
plt.title('Distribution of popularity for the genre Adventure over the years')
plt.axis([1960, 2015, 0, 3.5])
plt.show()


# In[33]:


df_new = df.groupby('release_year').mean()


# In[34]:


df_new['runtime'].describe()


# In[35]:


df_new['popularity'].hist(bins=36)
plt.xlabel('Popularity')
plt.ylabel('Counts')
plt.title('Popularity Over the Years');


# In[36]:


df_new['revenue'].hist(bins=24)
plt.xlabel('Revenue')
plt.ylabel('Counts')
plt.title('Revenue Over the Years');


# In[37]:


plt.scatter(x=df_new['revenue'], y=df_new['vote_average'])
plt.xlabel('Revenue')
plt.ylabel('Vote Averages')
plt.title('Revenue vs Vote Averages Over the Years');


# In[38]:


def find_top(dataframe_col, num=3):
    # split the characters in the input column 
    #and make it to a list
    alist = dataframe_col.str.cat(sep='|').split('|')
    #transfer it to a dataframe
    new = pd.DataFrame({'top' :alist})
    #count their number of appeared times and
    #choose the top3
    top = new['top'].value_counts().head(num)
    return top


# In[39]:



# sort the movie release year list.# sort th 
dfyear= df.release_year.unique()
dfyear= np.sort(dfyear)
dfyear


# In[40]:


y1960s =dfyear[:10]
# year list of 1970s
y1970s =dfyear[10:20]
# year list of 1980s
y1980s =dfyear[20:30]
# year list of 1990s
y1990s = dfyear[30:40]
# year list of afer 2000
y2000 = dfyear[40:]


# In[41]:


times = [y1960s, y1970s, y1980s, y1990s, y2000]
#generation name
names = ['1960s', '1970s', '1980s', '1990s', 'after2000']
#creat a empty dataframe,df_r3
df_r3 = pd.DataFrame()
index = 0
#for each generation, do the following procedure
for s in times:
    # first filter dataframe with the selected generation, and store it to dfn
    dfn = df[df.release_year.isin(s)] 
    #apply the find_top function with the selected frame, using the result create a dataframe, store it to dfn2 
    dfn2 = pd.DataFrame({'year' :names[index],'top': find_top(dfn.genres,1)})
     #append dfn2 to df_q2
    df_r3 = df_r3.append(dfn2)
    index +=1
df_r3


# In[42]:


generation = ['1960s', '1970s', '1980s', '1990s', 'after2000']
genres = df_r3.index
y_pos = np.arange(len(generation))
fig, ax = plt.subplots()
# Setting y1: the genre number
y1 = df_r3.top
# Setting y2 again to present the right-side y axis labels
y2 = df_r3.top
#plot the bar
ax.barh(y_pos,y1, color = '#007482')
#set the left side y axis ticks position
ax.set_yticks(y_pos)
#set the left side y axis tick label
ax.set_yticklabels(genres)
#set left side y axis label
ax.set_ylabel('genres')

#create another axis to present the right-side y axis labels
ax2 = ax.twinx()
#plot the bar
ax2.barh(y_pos,y2, color = '#27b466')
#set the right side y axis ticks position
ax2.set_yticks(y_pos)
#set the right side y axis tick label
ax2.set_yticklabels(generation)
#set right side y axis label
ax2.set_ylabel('generation')
#set title
ax.set_title('Genres Trends by Generation')


# In[43]:


The genre Drama are the most filmed in almost all generation. Only the 1980s are dominated by the comedy type.


# LIMITATIONS :
# 
# The medium through which popularity was determined is unknown. This can impact the analysis as the limitations and bias inherent while gauging audience response will be present in the end values too.
# 
# This analysis assumes that the same index and methods were employed for collecting popularity factors and counting votes for all movies. In the event that it is not so, the results might not hold true. 
# 
# While we did not have missing values for any of the factors under consideration, we acknowledge the presence of these limitations and assumptions in our analysis.

# In[ ]:


CONCLUSION:

Thus the most popular genre in most of the years is Drama. The above graphs show the popularity of each genre from 1960 to 2015. 
    

