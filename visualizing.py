from Cleaning_data     import cleaning_data
from pandas            import Series
from os.path           import join
from matplotlib.pyplot import show, subplots
from matplotlib.pyplot import style

style.use('seaborn-whitegrid')

ftsz_title_big = 27 + 10
ftsz_title     = 23 + 10
ftsz_label     = 20 + 10
ftsz_lgnd      = 18 + 10
linewidth      = 2

def visualize(datax, datay, num_plots = [4 , 2], y_label = "Converted", save_fig = False):

    ## We do not have that many variables, hence, we can actually plot everythingç
    ## and see which variables are interesting

    fig, ax = subplots(num_plots[0], num_plots[1], figsize=(30, 40))
    fig.suptitle(y_label,
                 fontsize = ftsz_title_big)

    i = 0
    for key in datax.keys():
        ax[i // 2, i % 2].scatter(datax[key], datay)
        ax[i // 2, i % 2].set_title(key, fontsize=ftsz_title)
        ax[i // 2, i % 2].set_ylabel(y_label, fontsize = ftsz_label)
        ax[i // 2, i % 2].set_xlabel(key, fontsize = ftsz_label)
        ax[i // 2, i % 2].tick_params(axis = 'both', labelsize = ftsz_lgnd)
        i += 1

    fig.subplots_adjust(bottom=0.05,
                top = 0.92,
                left=0.07,
                right=0.98,
                hspace = 0.2,
                wspace = 0.1
                )
    if save_fig:
        fig.savefig(join("Figures/", save_fig))
    return fig, ax

def plot_importance(ax, r, model_name, index):

    importances = Series(r.importances_mean, index=index)

    importances.plot.bar(yerr=r.importances_std, ax=ax)
    ax.set_title(f"Feature importances using permutation on {model_name}", fontsize = ftsz_title)
    ax.set_ylabel("Mean accuracy decrease", fontsize = ftsz_label)
    ax.tick_params(axis = 'both', labelsize = ftsz_lgnd)

    return ax

if (__name__ == "__main__"):

    path_to_dataset = "./"
    dataset_name = join(path_to_dataset, "Data_Scientist_-_Case_Dataset.xlsx")

    data = cleaning_data(dataset_name,
                        save_csv = False,
                        print_info = True)
    # segments = data["customer_segment"].unique()
    # for segment in segments:
    # data_sgmt = data.where(data['customer_segment'] != segment).dropna()
    data_sgmt = data.copy()
    convert = data_sgmt.pop('converted')
        # print(f"This data field has {data_sgmt.shape[1]} different parameters with {data_sgmt.shape[0]} entries")
    fig, ax = visualize(data_sgmt, convert, save_fig = f"Scatter.pdf")

    ## A part from visualizing we can show some percentage:

    female = data.loc[data['gender']]
    female_convert = female.converted.sum() / female.shape[0]
    male = data.loc[data['gender'] != 1]
    male_convert = male.converted.sum() / male.shape[0]
    print(f"{female_convert * 100:1.3f} % of the female users has converted")
    print(f"{male_convert * 100:1.3f} % of the male users has converted")

    cstm_sgmt_11 = data.loc[data['customer_segment'] == 0]
    cstm_sgmt_11_convert = cstm_sgmt_11.converted.sum() / cstm_sgmt_11.shape[0]
    cstm_sgmt_12 = data.loc[data['customer_segment'] == 1]
    cstm_sgmt_12_convert = cstm_sgmt_12.converted.sum() / cstm_sgmt_12.shape[0]
    cstm_sgmt_13 = data.loc[data['customer_segment'] == 2]
    cstm_sgmt_13_convert = cstm_sgmt_13.converted.sum() / cstm_sgmt_13.shape[0]
    print(f"{cstm_sgmt_11_convert * 100:1.3f} % of the segment 11 users has converted")
    print(f"{cstm_sgmt_12_convert * 100:1.3f} % of the segment 12 users has converted")
    print(f"{cstm_sgmt_13_convert * 100:1.3f} % of the segment 13 users has converted")


    branch_1 = data.loc[data['branch'] == 0]
    branch_1_convert = branch_1.converted.sum() / branch_1.shape[0]
    branch_2 = data.loc[data['branch'] == 1]
    branch_2_convert = branch_2.converted.sum() / branch_2.shape[0]
    branch_3 = data.loc[data['branch'] == 2]
    branch_3_convert = branch_3.converted.sum() / branch_3.shape[0]
    print(f"{branch_1_convert * 100:1.3f} % of the branch Helsinki users has converted")
    print(f"{branch_2_convert * 100:1.3f} % of the branch Tampere has converted")
    print(f"{branch_3_convert * 100:1.3f} % of the branch Turku has converted")
    show()
