# %%

#Changelog:
    #Made number of bins function of width of used image 
    #Added option to use different input image (read from csv file) and convert it to dictionary 
    #Made starting time a varibale rather than just using first entry
    #Repalced hardcoded pixel width in "visualize event_window" with variable to allow for plotting of data with different data dimensions
# =====================================================================================
# Import libraries
# =====================================================================================
# external libs
import numpy as np
import matplotlib.pyplot as plt
import rosbag
import tools.importing_dvs as i 


# =====================================================================================
# Define Functions
# =====================================================================================
def read_dvs_bag(bag_dir, topics=["raw_data"]):
    # open bag file
    bag = rosbag.Bag(bag_dir)

    x_t = []
    y_t = []
    pol_t = []
    t = []
    timestamps = []

    for _, msg, time in bag.read_messages(topics=topics):
        x = [event.x for event in msg.events]
        y = [event.y for event in msg.events]
        pol = [event.polarity for event in msg.events]
        ts = [event.ts for event in msg.events]

        x_t.append(x)
        y_t.append(y)
        pol_t.append(pol)
        t.append(ts)
        timestamps.append(time)

    bag.close()

    # store in dictionnary
    dvs_data = {}
    dvs_data["x"] = np.hstack(x_t)
    dvs_data["y"] = np.hstack(y_t)
    dvs_data["pol"] = np.hstack(pol_t)
    # convert genpy.Time to float array
    dvs_data["t"] = np.array([f.secs + f.nsecs * 10**(-9) for f in np.hstack(t)])
    dvs_data["timestamps"] = timestamps

    return dvs_data


def odd_even_fix(dvs_data):
    # count number of unique events at each x-address
    x_addr, n = np.unique(dvs_data["x"], return_counts=True)

    # construct temporary dict to store data (not rospy timestamps)
    # initialized to -1 to ease filtering
    corrected_data = {}
    for key in ["x", "y", "t", "pol"]:
        corrected_data[key] = -1 * np.ones_like(dvs_data[key])

    # loop through events by their x-event address
    for i in x_addr:
        # odd events (1, 3, ...)
        if i % 2 == 1:
            # find drop percentage to equalize it to the previous address
            p_drop = n[i - 1] / n[i]

            # select events at event address i
            valid = np.where(dvs_data["x"] == i)[0]

            # drop events as determined by p_drop
            #   multiply index j+1 (position in sequence of events with address i)
            #   with p_drop. If this changes by 1, an event is tagged as 'allowed'
            #   and stored.
            select = np.fromiter(
                (
                    x
                    for j, x in enumerate(valid)
                    if round((j + 1) * p_drop) - round(j * p_drop)
                ),
                dtype=int,
            )
        # even events (0, 2, ...)
        else:
            # if i is even select all events
            select = np.where(dvs_data["x"] == i)

        # store data into new dictionary at the selected indices
        for key in corrected_data.keys():
            corrected_data[key][select] = dvs_data[key][select]

    dvs_data_fixed = {}
    # select data > -1 to obtain address-corrected dvs data
    for key in corrected_data.keys():
        dvs_data_fixed[key] = corrected_data[key][corrected_data[key] > -1]

    return dvs_data_fixed


def visualize_event_window(ax, dvs_data, window, width):
    # visualize polarity
    colors = np.where(dvs_data["pol"], "r", "b")
    # select window of events
    window = np.where(np.logical_and(dvs_data["t"] >= t0, dvs_data["t"] <= t0 + dt))
    # plot
    ax.scatter(width -1 - dvs_data["x"][window], dvs_data["y"][window], s=0.1, color=colors[window])
    return

if __name__ == "__main__":
    # =====================================================================================
    # Load, correct and visualize dvs data
    # =====================================================================================

    roadmap = True
    rotating_disk = False
 
    
    if roadmap:
        height = 264 
        width = 320
        dir_event = '/home/yvonne/Documents/uni_stuff/thesis/code/cuSNN/cuSNN-samples/data/roadmap/final/final_0/events.csv'
        dvs_data = i.import_DVS(height,width).read_csv_data(dir_event, fix_stripes = False)
        dvs_data = dvs_data.to_dict('list')
        dvs_data = dict([key, np.array(dvs_data[key])] for key in ["x", "y", "t", "pol"])
        dvs_data_fixed = odd_even_fix(dvs_data)  # fix odd/even mismatch
        dt = 5000
      
    elif rotating_disk:
        height = 180
        width = 240
        dir_event = '/home/yvonne/Documents/uni_stuff/thesis/code/Thesis_SNN/data/disk/IMU_rotDisk/events.csv'
        dvs_data = i.import_DVS(height, width).read_csv_data(dir_event, fix_stripes = False)
        dvs_data = dvs_data.to_dict('list')
        dvs_data = dict([key, np.array(dvs_data[key])] for key in ["x", "y", "t", "pol"])
        dvs_data_fixed = odd_even_fix(dvs_data)  # fix odd/even mismatch
        dt = 5000
        

    else: 
        # %% Read and correct dvs data
        height = 180
        width = 240
        dvs_dir = "DVS_BAG.bag"  # directory of the dvs .bag file
        dvs_data = read_dvs_bag(dvs_dir)  # read dvs data
        dvs_data_fixed = odd_even_fix(dvs_data)  # fix odd/even mismatch
        dt = 0.03
        
        
        
    


    # %% Show difference between normal dvs data and corrected data

    # =====================================================================================
    # Visualize
    # =====================================================================================

    # Setup
    fig = plt.figure(figsize=(8, 12), dpi=100)

    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2, sharey=ax1)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4, sharey=ax3)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6, sharey=ax5)

    for a in [ax1, ax2, ax3, ax4, ax5, ax6]:
        a.set_xlim([0, 320])
        a.set_xlabel("x address [-]")

    # plot histogram
    bins = width
    ax1.set_title("Unaltered dvs data")
    ax1.hist(dvs_data["x"], bins = bins)
    ax1.set_ylabel("number of events [-]")

    ax2.set_title("Dvs data after odd/even fix")
    ax2.hist(dvs_data_fixed["x"], bins = bins)

    # plot buffered dvs windows and histogram for that window
    start_time = 200
    t0 = 1 + dvs_data["t"][start_time]
    #dt = 10000
    window = np.where(np.logical_and(dvs_data["t"] >= t0, dvs_data["t"] <= t0 + dt))

    visualize_event_window(ax3, dvs_data, window,width)
    ax3.set_ylabel("y address [-]")
    ax5.hist(dvs_data["x"][window], bins= bins)
    ax5.set_ylabel("number of events [-]")

    t0 = 1 + dvs_data_fixed["t"][start_time]
    #dt = 10000
    window = np.where(np.logical_and(dvs_data_fixed["t"] >= t0, dvs_data_fixed["t"] <= t0 + dt))

    visualize_event_window(ax4, dvs_data_fixed, window, width)
    ax6.hist(dvs_data_fixed["x"][window], bins = bins)
    plt.show()
