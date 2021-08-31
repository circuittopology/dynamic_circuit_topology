# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:31:47 2021

@author: scalvinib
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sy
import scipy.stats as ss
import scipy.signal
import pandas as pd



#Normalization function
def normalize(x):
    return (x - x.min(0)) / x.ptp(0)
    #normalize points

#Least squares fitting function        
def fit(x_dist,y_dist):    
    A= 7
    gamma=2
    guess=[A, gamma]

    errfunc2 = lambda p,x_dist, y: ( power_law(x_dist, *p) - y)**2
    optim, success= sy.optimize.leastsq(errfunc2, guess[:], args=( x_dist, y_dist))
    fit=power_law(x_dist, optim[0] , optim[1])
    return fit, optim

#Find peaks of a distribution
def find_peaks(length,n_peaks):
    plt.figure()
    ax = sns.kdeplot(length)
    x = ax.lines[0].get_xdata() # Get the x data of the distribution
    y = ax.lines[0].get_ydata() # Get the y data of the distribution
    peak= scipy.signal.find_peaks(y, threshold=0.00000)
    array=np.array(peak[0])
    #for t in range(len(array)):
        #plt.scatter(x[array[t]],y[array[t]])
    return x,array


#function which returns a SxT representation of the cmap array stack, 
#where S= NxN (N residues in the number of residues) and T is the number
#of MD frames. Suitable for a kymograph

def flatten_cmaps(start, end, start_frame, len_dataset, files_csv, path):
    dyn_cmap=np.zeros((end-start, end-start, len_dataset-start_frame))
    dyn_cmap_flat=np.zeros(((end-start)*(end-start), len_dataset-start_frame))
    for t in range(start_frame,len_dataset):
        cmap= pd.read_csv('{}/{}'.format(path, files_csv[t]), header=None)
        cmap=np.array(cmap)
        dyn_cmap[:,:,t-start_frame]=np.copy(cmap[start:end, start:end])
        dyn_cmap_flat[:,t-start_frame]= np.ndarray.flatten(np.copy(cmap[start:end, start:end]))
        graph_cmap=dyn_cmap_flat[~np.all(dyn_cmap_flat == 0, axis=1)]
    return graph_cmap, dyn_cmap_flat
    
#function which plots the kymograph of contacts
def plot_kymograph(graph_cmap, cmap=0):
    plot = plt.figure()
    plt.title('CONTACT LIFE SPAN, AR NTD DOMAIN')
    if cmap:
        plt.imshow(graph_cmap, aspect='auto', cmap= 'Blues')
    else:
        plt.imshow(graph_cmap, aspect='auto')
    plt.xlabel('Time (us)')
    plt.ylabel('Contact index')
    return plot


#Power law for lifetime distribution fit
def power_law(k, A, gamma):
    y= A*(k**(-gamma))
    return y


#Find the maximum lifetime per contact (pair of residues)
def find_max_lifetime_perres(dyn_cmap_flat, len_dataset):
    life_time_max=np.zeros((dyn_cmap_flat.shape[0], dyn_cmap_flat.shape[1]))

    for t in range(len_dataset):
        non_zero_contacts=np.array(np.nonzero(dyn_cmap_flat[:,t]))
        if t>0:
            for j in range(len(non_zero_contacts[0,:])):
                if (life_time_max[non_zero_contacts[0,j], t-1]==0):
                    life_time_max[non_zero_contacts[0,j],t]=1
                else:
                    life_time_max[non_zero_contacts[0,j],t]=life_time_max[non_zero_contacts[0,j],t-1]+1
            
        else:
            life_time_max[non_zero_contacts[0,:],t]=1
            
    life_time_max_arr = np.amax(life_time_max, 1)
    max_dist=np.copy(life_time_max_arr[life_time_max_arr != 0])
    return max_dist, life_time_max_arr* 0.005, life_time_max*0.005
    
#Create contact lifetime distribution    
def lifetime_distribution(max_dist):
    ax = sns.kdeplot(max_dist*0.005, bw_adjust=2)
    x = ax.lines[0].get_xdata() # Get the x data of the distribution
    y = ax.lines[0].get_ydata() # Get the y data of the distribution
    peak= scipy.signal.find_peaks(y, threshold=0.00000)
    array=np.array(peak[0])
    maxima=x[array]
    
    start_phys=20
    x=x[start_phys:-1]
    y=y[start_phys:-1]
    
    #Fit of the distribution
    fit_dist, optim=fit(x,y)
    print('Fit parameters:')
    print('A = {}, k = {}'.format(optim[0], optim[1]))
    
    return x,y,fit_dist,optim, maxima

#Plot lifetime distribution
def log_plot(x,y,fit_dist,optim):
    plot = plt.figure()
    plt.scatter(np.log(x),np.log(y))
    
    k="%.3f" % round(optim[1], 3)
    A="%.3f" % round(optim[0], 3)

    plt.title('CONTACT LIFETIME DISTRIBUTION')
    plt.plot(np.log(x), np.log(fit_dist), color='m', 
         label='P(t)= A*t^(-k)\nk={}, A={}'.format(k, A))
    
    thresh1=0.5
    thresh2=2
    plt.axvline(np.log(thresh2), 0, 1.5, color='c', ls= '--')
    plt.axvline(np.log(thresh1), 0, 1.5, color='c', ls= '--')
    plt.xlabel('Log(T (us))')
    plt.ylabel('Log(P(t))')
    plt.legend()
    return plot

#Function to save figures
def save_figures(plot,path, terminus, run, name_file):
    plot.savefig('{}/{}/{}/{}.jpg'.format(path, terminus, run, name_file))
    plot.savefig('{}/{}/{}/{}.eps'.format(path, terminus, run, name_file))
    
    
#Creates a lifetime filter for contact maps and saves it into a csv file
def create_mask(max_lifetime_array, life_filter, thresh, parameters, path_mask, save_mask=0):
    thresh1=thresh[0]
    thresh2=thresh[1]
    if (life_filter == 'long_life'):
        cut= np.copy(max_lifetime_array) 
        cut[cut<=thresh2]=0
    if (life_filter == 'middle_life'):
        cut=np.copy(max_lifetime_array)
        cut[cut<thresh1]=0
        cut[cut>thresh2]=0
    if (life_filter == 'short_life'):
        cut=np.copy(max_lifetime_array)
        cut[cut>thresh1]=0
        
    indexes_ravelled= np.nonzero(cut)
    x,y= np.unravel_index(indexes_ravelled, (parameters[3], parameters[3]), order='C')
    mask=np.zeros((parameters[3], parameters[3]))
    mask[x,y]=1
    
    mask_N = mask[parameters[0]:parameters[1], parameters[0]:parameters[1]]
    mask_C = mask[parameters[2]:parameters[3], parameters[2]:parameters[3]] 
        
    if save_mask:    
        np.savetxt('{}/{}_mask.csv'.format(path_mask, life_filter), mask,delimiter=",")
        np.savetxt('{}/{}_mask_N.csv'.format(path_mask, life_filter), mask_N,delimiter=",")
        np.savetxt('{}/{}_mask_C.csv'.format(path_mask, life_filter), mask_C,delimiter=",")    
    
    plt.figure()
    plt.title(life_filter)
    plt.imshow(mask, 'Blues_r')
    plt.xlabel('Residues')
    plt.ylabel('Residues')
    

#Function that computes all the lifetimes that each contact has during the MD run
def compute_lifetime(max_lifetime_array,lifetime_matrix):
    non_zero_index= np.array(np.nonzero(max_lifetime_array))
    non_zero_contacts=len(non_zero_index[0,:])
    life_times_allcont=[]

    for t in range(non_zero_contacts): 
        arr= lifetime_matrix[non_zero_index[0,t],:]    
        life_times=np.zeros(len(arr))
        switch=0
        time=0
        counter=0
        for j in range(len(arr)):
            if(switch==0 and arr[j]==0):
                switch=0
                time=0
            if(switch==0 and arr[j]!=0):
                time=arr[j]
                switch=1
            if(switch==1 and arr[j]!=0):
                time=arr[j]
            if(switch==1 and arr[j]==0):
                life_times[counter]=time
                counter=counter+1
                switch=0
                time=0
            if(switch==1 and arr[j]!=0 and j==len(arr)-1):
                life_times[counter]=time
                counter=counter+1
                switch=0
                time=0
      
        is_empty = life_times[life_times!=0]. size == 0.
        if (is_empty == True):
            print(t)
        life_times_allcont.append(life_times[life_times!=0])
    return life_times_allcont

#Function that calculates average lifetime per residue (C and C* factor)
def avg_lifetime_perres(all_lifetimes,max_lifetime_array, lifetime_matrix,C_termin_end):
    non_zero_index= np.array(np.nonzero(max_lifetime_array))
    mean_life_arr=np.zeros(len(all_lifetimes))
    for t in range(len(mean_life_arr)):
        life= np.array(all_lifetimes[t])
        if (len(life)!=1):
            mean_life_arr[t]=np.mean(life)
        else: 
            mean_life_arr[t]=life
    arr_mean=np.zeros(lifetime_matrix.shape[0])
    for t in range(mean_life_arr.shape[0]):
        arr_mean[non_zero_index[0,t]]=mean_life_arr[t]
        
    map_mean=np.zeros((C_termin_end,C_termin_end))
    indexes_ravelled= np.array(np.nonzero(arr_mean))
    
    for t in range(len(indexes_ravelled[0,:])):
        x,y= np.unravel_index(indexes_ravelled[0,t], (C_termin_end,C_termin_end), order='C')
        val=arr_mean[indexes_ravelled[0,t]]
        if val>0:
            map_mean[x,y]=val
      
    avg_per_residue=np.zeros(C_termin_end)
    for t in range(len(avg_per_residue)):
        line=map_mean[t,:]
        line=line[line>0.0]
        avg_per_residue[t]=np.mean(line)
        
    max_per_residue=np.zeros(C_termin_end)
    for t in range(len(max_per_residue)):
        line=map_mean[t,:]
        line=line[line>=0.0]
        max_per_residue[t]=np.max(line)
        
    return avg_per_residue, max_per_residue

#Plot C factors
def plot_Cfactors(avg_perres, max_perres, path_figures, run, savefig=0):
    a = plt.figure()
    plt.title('C FACTOR')
    plt.plot(avg_perres)
    plt.xlabel('Residue')
    plt.ylabel('Avg lifetime (us)')

    b = plt.figure()
    plt.title('C* factor')
    plt.plot(max_perres)
    plt.xlabel('Residue')
    plt.ylabel('Max lifetime (us)')
    
    if savefig:
        a.savefig('{}/all protein/{}/C_factor.jpg'.format(path_figures, run))
        b.savefig('{}/all protein/{}/Cstar_factor.jpg'.format(path_figures, run))
        a.savefig('{}/all protein/{}/C_factor.eps'.format(path_figures, run))
        b.savefig('{}/all protein/{}/Cstar_factor.eps'.format(path_figures, run))
    return 0
 
#Function to store C factor into a csv file
def save_cfactor(avg_perres, max_perres, path_Cfactor, run):
    c_factor={'C factor': avg_perres}
    Cfactor=pd.DataFrame(c_factor)
    Cfactor.to_csv('{}/{}/C_factor.csv'.format(path_Cfactor,run))

    cstar_factor={'Cstar factor': max_perres}
    Cstarfactor=pd.DataFrame(cstar_factor)
    Cstarfactor.to_csv('{}/{}/Cstar_factor.csv'.format(path_Cfactor,run))
    

#Function that calculates (average) contact range per residue, max contact range per residue and average contact
#range per residue weighted by contact lifetime.
def avg_contact_range(all_lifetimes,max_lifetime_array, lifetime_matrix,C_termin_end):
    non_zero_index= np.array(np.nonzero(max_lifetime_array))
    mean_life_arr=np.zeros(len(all_lifetimes)) 
    for t in range(len(mean_life_arr)):
        life= np.array(all_lifetimes[t])
        if (len(life)!=1):
            mean_life_arr[t]=np.mean(life)
        else: 
            mean_life_arr[t]=life
    arr_mean=np.zeros(lifetime_matrix.shape[0])
    for t in range(mean_life_arr.shape[0]):
        arr_mean[non_zero_index[0,t]]=mean_life_arr[t]
        
    map_mean=np.zeros((C_termin_end,C_termin_end))
    indexes_ravelled= np.array(np.nonzero(arr_mean))
    
    for t in range(len(indexes_ravelled[0,:])):
        x,y= np.unravel_index(indexes_ravelled[0,t], (C_termin_end,C_termin_end), order='C')
        val=arr_mean[indexes_ravelled[0,t]]
        if val>0:
            map_mean[x,y]=val
            
    range_per_residue=np.zeros(C_termin_end)
    for t in range(len(range_per_residue)):
        line=map_mean[t,:]
        line_indexes=np.nonzero(line)

        line_indexes=np.abs(line_indexes-(np.ones(len(line_indexes))*t))
        range_per_residue[t]=np.mean(line_indexes)
        
    max_range_per_residue=np.zeros(C_termin_end)
    for t in range(len(max_range_per_residue)):
        line=map_mean[t,:]
        line_indexes=np.nonzero(line)
        line_indexes=np.abs(line_indexes-(np.ones(len(line_indexes))*t))
        max_range_per_residue[t]=np.max(line_indexes)    

    weightrange_per_residue=np.zeros(C_termin_end)
    for t in range(len(range_per_residue)):
        line=map_mean[t,:]
        line_indexes=np.nonzero(line)
        line=line[line>0]

        line_indexes=np.abs(line_indexes-(np.ones(len(line_indexes))*t))
        weightrange_per_residue[t]=np.average(line_indexes[0,:], weights=line )

    return range_per_residue, max_range_per_residue, weightrange_per_residue


#Function that plots the spatial range of contacts. Three modes are available for the plot,'weighted', 'average',
# and 'max'.
def plot_contact_range(ranges, parameters, path_figures, run, mode, savefig = 0):
    
    if (mode == 'weighted'):
        contact_range= ranges[2]
    if (mode == 'max'):
        contact_range= ranges[1]
    if (mode == 'average'):
        contact_range= ranges[0]    
    
    range_N=np.copy(contact_range[parameters[0]:parameters[1]])
    range_C=np.copy(contact_range[parameters[2]:parameters[3]])
    
    plot= plt.figure()
    sns.histplot(range_N, color='pink', label= 'N terminus')
    sns.histplot(range_C, label= 'C terminus')
    
    plt.title('Contact range per residue, {}'.format(run))
    plt.xlabel('Average contact range (residues)')
    plt.legend()
    
    if savefig:
        plot.savefig('{}/all protein/{}/contact_range_{}.jpg'.format(path_figures, run, mode))
        plot.savefig('{}/all protein/{}/contact_range_{}.eps'.format(path_figures, run, mode))
    return plot

#Calculates the threshold for long range contacts
def calculate_threshold(ranges, mode):
    if (mode == 'weighted'):
        contact_range= ranges[2]
    if (mode == 'max'):
        contact_range= ranges[1]
    if (mode == 'average'):
        contact_range= ranges[0]    
    
    np.mean(contact_range)
    threshold= np.mean(contact_range)+np.std(contact_range)
    return threshold

#Calculates statistics for N terminus and C terminus contact range distribution
def distribution_stats_mode(ranges, parameters, mode):
    
    if (mode == 'weighted'):
        contact_range= ranges[2]
    if (mode == 'max'):
        contact_range= ranges[1]
    if (mode == 'average'):
        contact_range= ranges[0]    
    
    range_N=np.copy(contact_range[parameters[0]:parameters[1]])
    range_C=np.copy(contact_range[parameters[2]:parameters[3]])
    
    shapiro_N=ss.shapiro(range_N)
    shapiro_C=ss.shapiro(range_C)
    
    levene = ss.levene(range_N, range_C)
    
    mann= ss.mannwhitneyu(range_N, range_C)
    ttest= ss.ttest_ind(range_N, range_C)
    
    print('N terminus:')
    print(shapiro_N)
    print('C terminus:')
    print(shapiro_C)
    print('Comparison distributions:')
    print(levene)
    print(mann)
    print(ttest)
    
    return 0

#Set layout of the graphs
def set_layout(SMALL_SIZE = 13, MEDIUM_SIZE = 14, BIGGER_SIZE = 20):


    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    
    
#Arrange C factor from different runs in a dataframe
def create_df_cfactor(path_Cfactor, runs, c_factor):
    run_array=pd.DataFrame()
    
    for t in range(len(runs)):
        data=pd.read_csv('{}/{}/{}.csv'.format(path_Cfactor, runs[t], c_factor))
        #display(data)
        label = c_factor.replace("_", " ")
        col= data[label]
        nu_df={'{}'.format(runs[t]): col}
        nu_df=pd.DataFrame(nu_df)
        frames = [run_array, nu_df]
        run_array = pd.concat(frames,axis=1)
    return run_array
    
    
#Create plot above heatmap
def plot_over_map(data,avg, ax, ax_histx, c_factor):
    data=data.transpose()
    ax_histx.tick_params(axis="x", labelbottom=False)
    cmap = sns.color_palette("flare", as_cmap=True)
    sns.heatmap(data, cmap=cmap, ax=ax)
    label= c_factor.replace("_", " ")
    ax_histx.plot(avg, label='Average {}'.format(label))
    ax_histx.legend()
    return 0

#Creates plot made by two rectangles
def plot_rectangle(avg, C_factor_data, params, path_figures, c_factor, savefig = 0):
    #left, width, bottom, height, spacing,
    rect_scatter = [params[0], params[2], params[1], params[3]]
    rect_histx = [params[0], params[2] + params[3] + params[4], params[1]-0.22, 0.2]
   
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    plot_over_map(C_factor_data,avg, ax, ax_histx, c_factor)
    label= c_factor.replace("_", " ")
    label= label.upper()
    plt.title('{}, after 2 us'.format(label))
    
    if savefig:
        plt.savefig('{}/{}.eps'.format(path_figures, label),bbox_inches='tight')
        plt.savefig('{}/{}.jpg'.format(path_figures, label),bbox_inches='tight')
        
    return fig

#Read lifetime distributions and arrange them into a comprehensive dataframe
def read_lifetime_dist(path_distributions, runs, termini):
    distributions=pd.DataFrame()
    for t in range(len(runs)):
        for j in range(len(termini)):
            dist=pd.read_csv('{}/{}/{}/lifetime_distribution.csv'.format(path_distributions, termini[j], runs[t]))
            frames=[distributions, dist]
            distributions = pd.concat(frames,axis=0)

    distributions['Log Time(us)']=np.log(distributions['Time'])
    distributions['Log Distribution']=np.log(distributions['Lifetime distribution'])  
    return distributions

#Make comparison plot for contact lifetime distributions
def plot_comparison_dist(distributions, terminus, path_figures, savefig = 0):
    fig = plt.figure()
    sns.scatterplot(data=distributions[distributions['Terminus']==terminus], x="Log Time(us)", 
                    y="Log Distribution", hue="Run",linewidth=0.2)
    plt.axvline(np.log(2), 0, 1.5, color='c', ls= '--')
    plt.axvline(np.log(0.5), 0, 1.5, color='c', ls= '--')
    plt.title('Contact lifetime, {}'.format(terminus))
    if savefig:
        plt.savefig('{}/comparison_dist_{}.jpg'.format(path_figures, terminus))
        plt.savefig('{}/comparison_dist_{}.eps'.format(path_figures, terminus))        
    return fig
    
    
    
    
    
    
    
    
    
    
    