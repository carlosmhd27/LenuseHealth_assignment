from Cleaning_data import cleaning_data
from numpy import shape
from os.path import join, isfile, isdir
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, subplots,suptitle
from matplotlib.pyplot import style, legend

style.use('seaborn-whitegrid')

ftsz_title_big = 27 + 10
ftsz_title     = 23 + 10
ftsz_label     = 20 + 10
ftsz_lgnd      = 18 + 10
linewidth      = 2

def visualize(datax, datay, num_plots = [4, 2], y_label = "Converted", save_fig = False):

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
                    hspace = 0.98,
                    wspace = 0.1
                    )
    if save_fig:
        fig.savefig(join("./", save_fig))
    return fig, ax


if (__name__ == "__main__"):

    path_to_dataset = "./"
    dataset_name = join(path_to_dataset, "Data_Scientist_-_Case_Dataset_splitted.xlsx")

    data = cleaning_data(dataset_name,
                        save_csv = False,
                        print_info = True)
segments = data["customer_segment"].unique()
for segment in segments:
    data_sgmt = data.where(data['customer_segment'] != segment).dropna()
    convert = data_sgmt.pop('converted')
    print(f"This data field has {data_sgmt.shape[1]} different parameters with {data_sgmt.shape[0]} entries")
    fig, ax = visualize(data_sgmt, convert, save_fig = f"{segment}.pdf")
show()
