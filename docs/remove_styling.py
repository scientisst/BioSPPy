from notebooks._Utilities.styling import makePlain
import os


def formatAll(dir=None):
    """
    This function can be used to format several notebooks inside a folder.
    
    Parameters
    ----------
    dir: string
        Directory path of the folder containing the notebooks to format.
    """
    
    l_dir = get_all_filePaths(dir)
    for f_name in l_dir:
        if f_name.endswith('.ipynb'):
                path=f_name
                print(f_name)
                try:
                    makePlain(notebook=path,template=r"docs\notebooks\_Utilities\dictNB.txt")
                except:
                    print(f'Error in {f_name}')
                    pass

def get_all_filePaths(folderPath):
    result = []
    for dirpath, dirnames, filenames in os.walk(folderPath):
        result.extend([os.path.join(dirpath, filename) for filename in filenames])
    return result

formatAll(dir=r'C:\Users\sofia\Documents\GitHub\BioSPPy\docs\notebooks')