import numpy as np
import pandas as pd

# Basic Visualization tools
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_palette('husl')


# Bokeh (interactive visualization)
from bokeh.io import show, output_notebook
from bokeh.palettes import Spectral9
from bokeh.plotting import figure
output_notebook() # You can use output_file()

# Special Visualization
from wordcloud import WordCloud # wordcloud
import missingno as msno # check missing value

# Check file list
import os
os.chdir(r'c:\Users\DELL\Documents\project\VI')
print(os.listdir('fp'))

data = pd.read_csv('fp/appstore_games.csv')

#wordcloud
fig, ax = plt.subplots(1, 2, figsize=(16,32))
wordcloud = WordCloud(background_color='white',width=800, height=800).generate(' '.join(data['Name']))
wordcloud_sub = WordCloud(background_color='white',width=800, height=800).generate(' '.join(data['Subtitle'].dropna().astype(str)) )
ax[0].imshow(wordcloud)
ax[0].axis('off')
ax[0].set_title('Wordcloud(Name)')
ax[1].imshow(wordcloud_sub)
ax[1].axis('off')
ax[1].set_title('Wordcloud(Subtitle)')
plt.show()
#wordcloud

#regplot
fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)
sns.regplot(data=data, x='Price', y='Average User Rating', ax=ax)
plt.show()
#regplot

#heatmap
genre = data['Primary Genre'].value_counts()
p = figure(x_range=list(map(str, genre.index.values)), 
           plot_height=250, plot_width=1500, title="Primary Genre", 
           toolbar_location=None, 
           tools="")

p.vbar(x=list(map(str, genre.index.values)), 
       top=genre.values, 
       width=0.9, 
       color=Spectral9)

p.xgrid.grid_line_color = None
p.y_range.start = 0
show(p)

data['Genres'].head()

data['GenreList'] = data['Genres'].apply(lambda s : s.replace('Games','').replace('&',' ').replace(',', ' ').split()) 
data['GenreList'].head()

gameTypes = []
for i in data['GenreList']: gameTypes += i
gameTypes = set(gameTypes)
print("There are {} types in the Game Dataset".format(len(set(gameTypes))))


from sklearn.preprocessing import MultiLabelBinarizer # Similar to One-Hot Encoding

test = data['GenreList']
mlb = MultiLabelBinarizer()
res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_, index=test.index)

corr = res.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(15, 14))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
#heatmap


#pricePaid
data['Original Release Date'] = pd.to_datetime(data['Original Release Date'], format = '%d/%m/%Y')
date_size = pd.DataFrame({'size':data['Size']})
date_size = date_size.set_index(data['Original Release Date'])
date_size = date_size.sort_values(by=['Original Release Date'])
date_size.head()

date_size['size'] = date_size['size'].apply(lambda b : b//(2**10)) # B to KB

monthly_size = date_size.resample('M').mean()
tmp = date_size.resample('M')
monthly_size['min'] = tmp.min()
monthly_size['max'] = tmp.max()
monthly_size.head()

fig = figure(x_axis_type='datetime',           
             plot_height=250, plot_width=750,
             title='Date vs App Size')
fig.line(y='size', x='Original Release Date', source=date_size)
show(fig)

paid = data[data['Price']>0]
free = data[data['Price']==0]
fig, ax = plt.subplots(1, 2, figsize=(15,8))
sns.countplot(data=paid, y='Average User Rating', ax=ax[0], palette='plasma')
ax[0].set_title('Paid Games')
ax[0].set_xlim([0, 1000])
#pricePaid

sns.countplot(data=free, y='Average User Rating', ax=ax[1], palette='viridis')
ax[1].set_title('Free Games')
ax[1].set_xlim([0,2000])
plt.tight_layout();

import io
import base64

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)

    return base64.b64encode(img.getvalue())

encoded = fig_to_base64(fig)
my_html = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))

def convert_fig_to_html(fig):
  """ Convert Matplotlib figure 'fig' into a <img> tag for HTML use using base64 encoding. """
  import urllib
  from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
  import StringIO
  canvas = FigureCanvas(fig)
  png_output = StringIO.StringIO()
  canvas.print_png(png_output)
  return '<img src="data:image/png;base64,{}">'.format(urllib.quote(png_output.getvalue().encode('base64').rstrip('\n')))