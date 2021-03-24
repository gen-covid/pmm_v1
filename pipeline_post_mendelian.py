import numpy as np
import pandas as pd
import copy
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as pl
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from mord import LogisticAT

mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.labelsize'] = 14

def metrics(model, X, y, sample_weight=None, fit_params=None, random_state=0, n_jobs=4):
    list_score = ['accuracy', 'precision', 'recall', 'balanced_accuracy', 'roc_auc'] 
    cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    df_cv = pd.DataFrame(cross_validate(model, X, y, cv=cv, scoring=list_score, fit_params=fit_params))[['test_'+tt for tt in list_score]]
    y_pred = cross_val_predict(model, X, y, cv=cv, fit_params=fit_params, n_jobs=n_jobs)
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm, index=['Negative', 'Positive'], columns=['Negative (predicted)', 'Positive (predicted)'])
    pl.figure(str(model.__class__) + '_confusion_matrix')
    sns.set(font_scale=2)
    sns.heatmap(cm_df, center=0, cmap=sns.diverging_palette(220, 15, as_cmap=True), annot=True, fmt='g')
    pl.xticks(rotation=0)
    pl.yticks(rotation=90, va="center")
    mng = pl.get_current_fig_manager()
    mng.window.showMaximized()
    pl.show()
    df_overall = df_cv.mean().to_frame(model.__class__)
    return df_cv, df_overall

# set paths, param, analysis
base_folder = ''
data_folder = base_folder + ''
saving_folder = base_folder + ''
number_samples = ''
fold = 10
num_trials = 3
analysis = ['AD', 'male', 'X']
#the different analyses that can be performed are:
#[AD, male], [AR, male], [GC, male], [GC_HOMO, male], ['AD', 'male', 'X'], ['GC', 'male', 'X']
#[AD, female], [AR, female], [GC, female], [GC_HOMO, female]


# load df_task
df_task_original = pd.read_csv(data_folder + 'phenotype.csv').set_index('sample')[['age', 'gender', 'grading']]
df_task_original = df_task_original.dropna()
# OLR
model_ordinal = LogisticAT(alpha=0)   
model_ordinal_m = copy.deepcopy(model_ordinal)
df_task_original_m = df_task_original[df_task_original['gender']==0]
model_ordinal_m.fit(df_task_original_m[['age']].astype(int), df_task_original_m['grading'].astype(int))
y_pred_m = model_ordinal_m.predict(df_task_original_m[['age']])
df_task_original.loc[df_task_original_m.index, 'ordered_LR_prediction_sex'] = y_pred_m
model_ordinal_f = copy.deepcopy(model_ordinal)
df_task_original_f = df_task_original[df_task_original['gender']==1]
model_ordinal_f.fit(df_task_original_f[['age']].astype(int), df_task_original_f['grading'].astype(int))
y_pred_f = model_ordinal_f.predict(df_task_original_f[['age']])
df_task_original.loc[df_task_original_f.index, 'ordered_LR_prediction_sex'] = y_pred_f
df_task_original['delta_sex'] = df_task_original['grading'] - df_task_original['ordered_LR_prediction_sex']

pl.figure()
df_plot_m = df_task_original[df_task_original['gender']==0]
y_pred_plot_m = model_ordinal_m.predict(df_plot_m[['age']])
pl.subplot(2, 1, 1)
pl.title('male')
df_plot_m['ordered_LR_prediction'] = y_pred_plot_m
df_plot_m['Patient distribution'] = 'actual distribution'
df_task_original_copy = copy.deepcopy(df_plot_m)
df_task_original_copy['grading'] = y_pred_plot_m
df_task_original_copy['Patient distribution'] = 'predicted distribution'
df_task_ = pd.concat([df_plot_m[['age', 'grading', 'Patient distribution']], df_task_original_copy[['age', 'grading', 'Patient distribution']]], 0)
sns.violinplot(x='age', y='grading', data=df_task_, orient='h', inner=None, hue='Patient distribution', split=True,
               order = [4, 3, 2, 1, 0], color='gray', scale='count')
pl.plot(df_plot_m['age'], 4-df_plot_m['grading'], 'or')
pl.plot(df_plot_m['age'], 4-y_pred_plot_m, 'ok', label='predicted (OLR)')
ok = df_plot_m[df_plot_m['delta_sex']<0]
pl.plot(ok['age'], 4-ok['grading'], 'og')
pl.ylabel('grading')
pl.legend()

df_plot_f = df_task_original[df_task_original['gender']==1]
y_pred_plot_f = model_ordinal_f.predict(df_plot_f[['age']])
pl.subplot(2, 1, 2)
pl.title('female')
df_plot_f['ordered_LR_prediction'] = y_pred_plot_f
df_plot_f['Patient distribution'] = 'actual distribution'
df_task_original_copy = copy.deepcopy(df_plot_f)
df_task_original_copy['grading'] = y_pred_plot_f
df_task_original_copy['Patient distribution'] = 'predicted distribution'
df_task_ = pd.concat([df_plot_f[['age', 'grading', 'Patient distribution']], df_task_original_copy[['age', 'grading', 'Patient distribution']]], 0)
sns.violinplot(x='age', y='grading', data=df_task_, orient='h', inner=None, hue='Patient distribution', split=True,
               order = [4, 3, 2, 1, 0], color='gray', scale='count')
pl.plot(df_plot_f['age'], 4-df_plot_f['grading'], 'or', label='actual')
pl.plot(df_plot_f['age'], 4-y_pred_plot_f, 'ok', label='predicted (OLR)')
ok = df_plot_f[df_plot_f['delta_sex']<0]
pl.plot(ok['age'], 4-ok['grading'], 'og', label='actual')
pl.ylabel('grading')
pl.xlabel('age')
pl.legend('', frameon=False)

def f_(t):
    if t in grading_positive:
        return 1
    elif t in grading_negative:
        return 0
    else:
        return 'none'

goal = 'delta_grading'
grading_positive = [1, 2, 3, 4]
grading_negative = [-1, -2, -3, -4]
df_task_original['delta_grading'] = df_task_original['delta_sex'].apply(f_)

# dataset RARE
df_original = pd.DataFrame()

if 'AD' in analysis:
    df_original = pd.read_csv(data_folder + 'data_al1_rare.csv').set_index('Unnamed: 0')
    col_name = []
    for gene_ in df_original.columns:
        col_name.append(gene_ + '_rare_al1')
    df_original.columns = col_name

elif 'AR' in analysis:
    df_original = pd.read_csv(data_folder + 'data_al2_rare.csv').set_index('Unnamed: 0')
    col_name = []
    for gene_ in df_original.columns:
        col_name.append(gene_ + '_rare_al2')
    df_original.columns = col_name

# dataset GC
elif 'GC' in analysis:
    df_original = pd.read_csv(data_folder + 'data_gc_hetero.csv').set_index('Unnamed: 0')

elif 'GC_HOMO' in analysis:
    df_original = pd.read_csv(data_folder + 'data_gc_homo.csv').set_index('Unnamed: 0')

if 'X' in analysis:
    df_chrx = pd.read_csv('list_genes_chrX.csv')['genes_x'].values
    to_excl_non_x = []            
    for col_df in df_original.columns: 
        if col_df.split('_')[0] not in df_chrx.tolist():
            to_excl_non_x.append(col_df)
    to_excl_non_x = list(set(to_excl_non_x))
    df_original = df_original.drop(to_excl_non_x, axis=1)

# target variable
d = {'M': 1, 'F': 0}
df_task_original['male'] = 1 - df_task_original['gender']
df_original = pd.merge(left=df_original, right=df_task_original['male'], how='inner', left_index=True, right_index=True)
if 'male' in analysis:
    df_original = df_original[(df_original['male']==1)]
if 'female' in analysis:
    df_original = df_original[(df_original['male']==0)]
del df_original['male']
assert df_original.isna().sum().sum() == 0
df = copy.deepcopy(df_original)  
task = []
for ind in df.index:
    target = df_task_original.loc[ind][goal]    
    if target == 'none':
        task.append(np.nan)
    elif target in [0]:
        task.append(0)
    elif target in [1]:
        task.append(1)
    else:
        print('error', target)
            
df = df[~np.isnan(task)]
task_no_nan = np.array(task)[~np.isnan(task)]

# preprocessing
to_excl_gene = []
for jj in df.columns:
    if (1 == df[jj]).all() or (0 == df[jj]).all():
        to_excl_gene.append(jj)
df = df.drop(to_excl_gene, axis=1)
df_clean = df.T.reset_index().rename(columns={'index':'complete'})
n_samples = df.shape[0]
n_features = df.shape[1]
y = np.array(task_no_nan)
X = df.values
print('dataset dimension: ', X.shape)
X_train, y_train = shuffle(X, y, random_state=0)

# grid search
param = np.logspace(np.log10(1e-2), np.log10(1e3), 51).tolist()
parameters = {'C':param}
logreg = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', max_iter=500)
matrix_score = np.zeros((num_trials, len(param)))
matrix_score_std = np.zeros((num_trials, len(param)))
for i in range(num_trials):
    cv_i = StratifiedKFold(n_splits=fold, shuffle=True, random_state=i)
    clf = GridSearchCV(logreg, parameters, cv=cv_i, scoring='roc_auc')
    clf.fit(X_train, y_train)
    matrix_score[i, :] = clf.cv_results_['mean_test_score']
    matrix_score_std[i, :] = clf.cv_results_['std_test_score']
scores_lr_cv = np.mean(matrix_score, 0)
scores_std_lr = np.mean(matrix_score_std, 0)
highest_param_lr = np.max(scores_lr_cv) - scores_std_lr[np.argmax(scores_lr_cv)]/2.
for par, sco in zip(param, scores_lr_cv):
    if sco >= highest_param_lr:
        best_param_lr = par
        break

# Model fitting
logreg_model = LogisticRegression(penalty='l1', solver='liblinear', C=best_param_lr)

# Cross-validation
scores_lr, score_lr = metrics(logreg_model, X_train, y_train, random_state=0)

# performances
pl.figure()
scores_lr['test_balanced_accuracy'] = 2*scores_lr['test_balanced_accuracy'] - scores_lr['test_recall']
ax = scores_lr.boxplot(showmeans=True)
pl.ylim(0, 1)
ax.set_xticklabels(['accuracy', 'precision', 'sensitivity', 'specificity', 'roc-auc'])

# feature Rank
logreg_model.fit(X_train, y_train)
logreg_model.predict(X_train) - y_train
df_res = pd.DataFrame(copy.deepcopy(df_clean['complete']))
df_res['Feature Importance'] = logreg_model.coef_[0]
df_res['abs_score'] = np.abs(logreg_model.coef_[0])
df_fin_logreg = copy.deepcopy(df_res.sort_values(['abs_score'], ascending=False))
df_fin_logreg.rename(columns={'complete':'log regression'}, inplace=True)
df_plot = df_res.sort_values(['abs_score'], ascending=False)
ax = df_plot[df_plot['abs_score']>0].plot.bar(x='complete', y='Feature Importance', rot=270)
mng = pl.get_current_fig_manager()
pl.xticks(fontsize=12)
pl.rcParams['font.size'] = 2

# Grid search plot
pl.figure('cross validation logistic regression_ '+ goal )
pl.errorbar(1./np.array(param), scores_lr_cv, scores_std_lr/2., marker='o', linestyle='-'); pl.xscale('log')
pl.plot(1./best_param_lr, sco, marker='o', color='r', markersize=12)
pl.ylabel('cross-validation score')
pl.xlabel('LASSO parameter')

