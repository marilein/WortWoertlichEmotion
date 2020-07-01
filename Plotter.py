import seaborn as sns
import data_processor as dtp


base_path = 'raw_data/final_data/'
experiment_files = dtp.get_filelist_in_folder(base_path)
