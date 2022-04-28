################
""" 
author: benedikt.seitzer
name: module_palm_pyplot
purpose: plot and process PALM-Data
"""
################

################
"""
IMPORTS
"""
################
import os

################
"""
FUNCTIONS
"""
################

__all__ = [
    'prepare_plotfolder'
]

# os-file-system operations
def prepare_plotfolder(run_name,run_number):
    """
    check if outputfolder for plots is there. If not, create and prepare it 

    -----------
    Parameters
    run_name: str
    run_number: str

    """

    if run_number == '':
        run_number = '.000'

    path_sim = '../palm_results/{}'.format(run_name) 
    path = '../palm_results/{}/run_{}'.format(run_name,run_number[-3:])
    path_cross = '{}/crosssections'.format(path)
    path_profile = '{}/profiles'.format(path)
    path_times = '{}/timeseries'.format(path)
    path_spectra = '{}/spectra'.format(path)
    path_lux = '{}/lux'.format(path)
    path_turbint = '{}/turbint'.format(path)    
    path_palminput = '../palm/current_version/JOBS/{}/INPUT'.format(run_name) 

    # initial 
    if os.path.exists('../palm_results/'):
        print('\n ../palm_results/ already exists \n')
    else:
        os.mkdir('../palm_results/')

    if os.path.exists(path_sim):
        print('\n project-directory already exists \n')
    else:
        os.mkdir(path_sim)

    if os.path.exists(path):
        print('\n all output-folders exist \n')
    else:
        try:
            # creates directories
            os.mkdir(path)
            os.mkdir(path_cross)
            os.mkdir(path_profile)
            os.mkdir(path_times)
            os.mkdir(path_spectra)
            os.mkdir(path_lux)
            os.mkdir(path_turbint)
            # copies INPUT-file of PALM-run to path
            os.system('cp {}/{}_p3d {}'.format(path_palminput,run_name,path))
            try:
                os.system('cp {}/{}_p3dr {}'.format(path_palminput,run_name,path))            
            except:
                print('no restart file')
            try:
                os.system('cp {}/{}_topo {}'.format(path_palminput,run_name,path))            
            except:
                print('no topo file')
        except OSError:
            print ('\n Creation of the directories {} failed \n'.format(path))
        else:
            print ('\n Successfully created the directories {} \n'.format(path))
