import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("Hb.csv")

x=df[['GEIM800106','GEIM800110','PUNT030102','SWER830101','OOBM850101','CORJ870107','GEIM800107','CORJ870102','QIAN880120','QIAN880131','IC50']]

#data1 = x.corr(method ='pearson')


######################## THE ONE ##################################

"""data1 = df.corr(method ='spearman')
onlyOne = data1[['IC50']] >= 0.27 #>= 0.25 #and <=-0.25 for specific selection
#onlyOne = data1[['IC50']] <= -0.3
#print(data1)
print(onlyOne)"""

######################## THE ONE ##################################
corrmat = x.corr(method ='pearson')
cg = sns.clustermap(corrmat, cmap ="cool", linewidths = 0.1, annot=True); #rainbow
plt.figure(figsize=(120,100)) #120,100
print(plt.show())
"""#plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0)
#sns.heatmap(df.corr(), annot=True)"""
"""plt.figure(figsize=(9,5))
sns.heatmap(df.corr(),annot=True, cmap ="rainbow")"""


"""cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]"""