import pandas as pd
import copy 
import os
import re
from collections import OrderedDict
import itertools

# Global variables
sub_metrics = ['Err ($\\downarrow$)', 'Cov ($\\uparrow$)']
### Name mapping
# Data-model identifiers
dm_identifier = {
                'mnist': 'MNIST', 
                'cifar10': 'CIFAR-10', 
                'twenty_newsgroups': '20 Newsgroups',
                'tiny_imagenet': 'Tiny-ImageNet'
                }
# Calibration method identifers
cms_ = OrderedDict({'None': 'Softmax',
                    'temp_scaling': 'TS',
                    'dirichlet': 'Dirichlet',
                    'scaling_binning': 'SB',
                    'histogram_binning_top_label': 'Top-HB',
                    'auto_label_opt_v0': 'Ours'
                    })
# Train-time method identifiers
ttms_ = OrderedDict({'std_cross_entropy': 'Vanilla',
                     'crl': 'CRL', 
                     'fmfp': 'FMFP', 
                     'squentropy': 'Squentropy'})

# Output file name
output_file_name = "./final_table_latex_template.txt"
# Directory path to read
directory_path_to_read = "../outputs/final_results/final_results_to_tex_table"
# directory_path_to_read = "./all_eval_full_xlsx"
# Decimal places
num_dp = 1
# Font sizes
global_font_size = (10,11)
std_font_size = (7, 11)
pm_factor = 0.6 # plus minus symbol factor size 
section_space_height = "4pt"
row_space_height = "2pt"
header_space_height = "2pt"
# General utils
visited = []
body_txt= ""
bs = "\\"
caption = "In every round the error was enforced to be below 5\%; TS stands for Temperature Scaling, SB stands for Scaling Binning, Top-HB stands for Top-Label Histogram Binning"


# This function traverses a directory and reads all xlsx files into dataframe
def helper_find_files_and_read_dataframe(root_path, patterns=[r"cifar10"]):
    dm_df= OrderedDict((dm, None) for dm in dm_identifier.keys()) 
    for root, _, files in os.walk(root_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            if os.path.isfile(filepath) and filepath.endswith(".xlsx"):
                for pattern in patterns:
                    if re.search(pattern, filename, re.IGNORECASE):
                        if pattern in dm_df:
                            dm_df[pattern] = pd.concat([dm_df[pattern], pd.read_excel(filepath, sheet_name=0).drop(columns=['Unnamed: 0'])], axis=0)
                        else:
                            dm_df[pattern] = pd.read_excel(filepath, sheet_name=0).drop(columns=['Unnamed: 0']).copy(deep=True)
                        break 
    return dm_df 

# This function filters the dataframes that are stored
def read_and_get_filtered_dataframes(root_path, patterns=[r"cifar10"]):
    dm_df = helper_find_files_and_read_dataframe(root_path, patterns = patterns)
    # Apply filter to all dataframes
    for dm, df in dm_df.items():
        if df is None:
            continue
        df1 = copy.copy(df) # Shallow copy to new dataframe
        df1['calib_conf'] = df1['calib_conf'].fillna("None")
        df1['calib_conf'] = df1['calib_conf'].astype(str)
        # Sort by col: Coverage-Mean in descending order, and then by col: calib_conf in ascending order
        df2 = df1.sort_values(["Coverage-Mean", "calib_conf"], ascending = [False, True]).copy(deep=True)
        # Retain the first row for each unique value in col: calib_conf
        df3 = df2.drop_duplicates(subset=['calib_conf', 'training_conf'], keep='first').copy(deep=True)
        dm_df[dm] = df3
    return dm_df 

# Main run
if __name__ == "__main__":
    dm_df = read_and_get_filtered_dataframes(
        root_path = f"{directory_path_to_read}", 
        patterns = dm_identifier.keys())

    counter = 0
    # LaTeX string formatting logic
    for tm, cm in itertools.product(ttms_.keys(), cms_.keys()):
        cross_prod_i = cms_[cm]
        # Bold our method
        cross_prod_i = f"\\textbf{{{cross_prod_i}}}" if cross_prod_i == "Ours" else cross_prod_i
        if tm not in visited:
            temp_tm = ttms_[tm]
            cross_prod_i = rf"""\multirow{{6}}{{*}}{{{temp_tm}}}                     & """ + cross_prod_i 
            visited.append(tm)
        else:
            cross_prod_i = " ".join(["                                 & ", cross_prod_i]) 
        for dm, df in dm_df.items():
            if df is not None:
                mask1 = (df["calib_conf"] == f"{cm}") & (df["training_conf"] == f"{tm}")
                al_mean = df[mask1]['Auto-Labeling-Err-Mean'].values[0] if not df[mask1]['Auto-Labeling-Err-Mean'].empty else -1 
                al_std = df[mask1]['Auto-Labeling-Err-Std'].values[0] if not df[mask1]['Auto-Labeling-Err-Std'].empty else -1 
                c_mean = df[mask1]['Coverage-Mean'].values[0] if not df[mask1]['Coverage-Mean'].empty else -1 
                c_std= df[mask1]['Coverage-Std'].values[0] if not df[mask1]['Coverage-Std'].empty else -1 

                # Bold the min al_mean and max c_mean 
                mask2 = (df["training_conf"] == f"{tm}")
                if al_mean == df[mask2]['Auto-Labeling-Err-Mean'].min():
                    al_mean_str_format, al_std_str_format = (rf"\textbf" + "{" + rf"{al_mean:.{num_dp}f}" + "}", rf"\textbf" + "{" + rf"{al_std:.{num_dp}f}" + "}")
                    al_plus_minus = f"\\scalebox{{{pm_factor}}}{{\\ensuremath{{\\bm{{\\pm}}}}}}"
                else:
                    al_mean_str_format, al_std_str_format = (rf"{al_mean:.{num_dp}f}", rf"{al_std:.{num_dp}f}")
                    al_plus_minus = f"\\scalebox{{{pm_factor}}}{{\\ensuremath{{{bs}pm}}}}"

                if c_mean == df[mask2]['Coverage-Mean'].max():
                    c_mean_str_format, c_std_str_format = (rf"\textbf" + "{" + rf"{c_mean:.{num_dp}f}" + "}",rf"\textbf" + "{" + rf"{c_std:.{num_dp}f}" + "}")
                    c_plus_minus = f"\\scalebox{{{pm_factor}}}{{\\ensuremath{{\\bm{{\\pm}}}}}}"
                else: 
                    c_mean_str_format, c_std_str_format = (rf"{c_mean:.{num_dp}f}", rf"{c_std:.{num_dp}f}")
                    c_plus_minus = f"\\scalebox{{{pm_factor}}}{{\\ensuremath{{{bs}pm}}}}"
            else:
                al_mean, al_std, c_mean, c_std = -1, -1, -1, -1 
            open_std_font = "{" + f"{bs}fontsize{{{std_font_size[0]}}}{{{std_font_size[1]}}}{bs}selectfont"
            closing_std_font = "}" 

            plus_minus = f"\\scalebox{{{pm_factor}}}{{\\ensuremath{{{bs}pm}}}}"

            # Add color only for error
            cross_prod_i = cross_prod_i + " & " + f""" \cellcolor{{red!10}}{ al_mean_str_format + al_plus_minus + open_std_font + al_std_str_format} """ + closing_std_font + " & " + f""" { c_mean_str_format + c_plus_minus + open_std_font + c_std_str_format} """ + closing_std_font

            # Add color only for error and coverage
            # cross_prod_i = cross_prod_i + " & " + f""" \cellcolor{{red!10}}{ al_mean_str_format + al_plus_minus + open_std_font + al_std_str_format} """ + closing_std_font + " & " + f""" \cellcolor{{emerald!10}}{ c_mean_str_format + c_plus_minus + open_std_font + c_std_str_format} """ + closing_std_font

            # No color
            # cross_prod_i = cross_prod_i + " & " + f""" { al_mean_str_format + al_plus_minus + open_std_font + al_std_str_format} """ + closing_std_font + " & " + f""" { c_mean_str_format + c_plus_minus + open_std_font + c_std_str_format} """ + closing_std_font

        if cm == list(cms_.keys())[-1] and tm == list(ttms_.keys())[-1]:
            # Add a bottom rule to the last row
            #line = rf"\noalign{{\vskip{section_space_height}}} \bottomrule"
            line = rf"\noalign{{\vskip{section_space_height}}} \bottomrule"
        elif cm == list(cms_.keys())[-1] and tm != list(ttms_.keys())[-1]:
            # Add horizontal line between train-time methods
            #line = f"\\noalign{{\\vskip{section_space_height}}} \\hline \\noalign{{\\vskip{section_space_height}}}"
            line = f"\\noalign{{\\vskip{section_space_height}}} \\hline \\noalign{{\\vskip{section_space_height}}}"

        else:
            #line = rf"\noalign{{\vskip{row_space_height}}}" + f"\hhline{{~---------}}" + rf"\noalign{{\vskip{row_space_height}}}"
            line = rf"\noalign{{\vskip{row_space_height}}}" + f"\hhline{{~---------}}" 
        cross_prod_i = cross_prod_i + r"\\" + line + rf"\noalign{{\vskip{row_space_height}}}" 
        #body_txt= body_txt+ cross_prod_i + "\n" 
        body_txt= body_txt+ cross_prod_i 

    # String formatting tools
    # Adding color/ shading to the "Error" column
    metrics_parts = []
    for _ in dm_identifier.keys():
        for sm in sub_metrics:
            if sm == "Err ($\\downarrow$)":
                # With color
                part = rf" \multicolumn{{1}}{{c}}{{\cellcolor{{red!20}}\textbf{{{sm}}}}}"
                # No color
                # part = rf" \multicolumn{{1}}{{c}}{{\textbf{{{sm}}}}}"
            else:
                # With color
                # part = rf" \multicolumn{{1}}{{c}}{{\cellcolor{{emerald!20}}\textbf{{{sm}}}}}"
                # No color
                part = rf"\multicolumn{{1}}{{c}}" + "{" + rf"\textbf{{{sm}}}" + "}"
            metrics_parts.append(part)
    metrics_txt = " & ".join(metrics_parts)

    data_models_txt = ' & ' + ' & '.join([rf"\multicolumn{{{len(sub_metrics)}}}{{c}}" + "{" + rf"\textbf" + rf"{{{dm_identifier[dm]}}}" + "}" for dm in dm_identifier.keys()])

    template = rf"""
    \begin{{table*}}[t]
    \centering
    \fontsize{{{global_font_size[0]}}}{{{global_font_size[1]}}}\selectfont
    \begin{{tabular}}{{llcccccccc}}
    \toprule \noalign{{\vskip{header_space_height}}}
    \multicolumn{{1}}{{c}}{{\multirow{{2}}{{*}}{{\textbf{{Train-time}}}}}} & \multicolumn{{1}}{{c}}{{\multirow{{2}}{{*}}{{\textbf{{Post-hoc}}}}}} {data_models_txt} \\ \noalign{{\vskip{header_space_height}}} \cline{{3-10}} \noalign{{\vskip{header_space_height}}} 
    \multicolumn{{1}}{{c}}{{}}                      & \multicolumn{{1}}{{c}}{{}}  & {metrics_txt} \\ \noalign{{\vskip{header_space_height}}} \toprule \noalign{{\vskip{section_space_height}}} 
    """ + body_txt + rf"""
    \end{{tabular}}
    \caption{{{caption}}}
    \label{{table:final_table}}
    \end{{table*}}"""

    # Write to text file
    with open(f"{output_file_name}", "w") as file:
        file.write(template)

