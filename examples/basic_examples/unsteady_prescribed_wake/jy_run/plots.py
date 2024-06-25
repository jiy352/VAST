import numpy as np

def axis_equal(ax, x, y, z):
    # Calculate the center and the range of the data in each dimension
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5

    # Set the limits so that the range is the same and centered around the data
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  


def plot3d(ax, mesh_all_ts, panel_velocity_all_ts, make_gif=True):
    import matplotlib.pyplot as plt
    import time
    num_time_steps = mesh_all_ts.shape[0]
    filenames = []
    for i in range(num_time_steps):
        ax.clear()  # Clear previous points
        axis_equal(ax, mesh_all_ts[:,:,:,0], mesh_all_ts[:,:,:,1], mesh_all_ts[:,:,:,2])
        data = mesh_all_ts[i]
        x = data[:,:,0].flatten()
        y = data[:,:,1].flatten()
        z = data[:,:,2].flatten()

        ax.scatter(x, y, z)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # Scatter plot for the mesh points
        ax.scatter(x, y, z, color='b')

        # Adding velocity arrows
        panel_center = (0.25*(data[0:-1, 0:-1, :] + data[0:-1, 1:, :] + data[1:, 0:-1, :] + data[1:, 1:, :])).reshape(-1, 3)

        if panel_velocity_all_ts is not None:
            

            velocity_data = panel_velocity_all_ts[i]
            u = velocity_data[:, :, 0].flatten()  # Velocity components in x
            v = velocity_data[:, :, 1].flatten()  # Velocity components in y
            w = velocity_data[:, :, 2].flatten()  # Velocity components in z


            ax.quiver(panel_center[:,0], panel_center[:,1], panel_center[:, 2],
                            u, v, w, color='r')

        plt.draw()
        plt.pause(0.01)  # Pause to update the plot
        time.sleep(0.01)  # Adjust as per your time step's actual duration

        # Save the current frame
        filename = f'frame_{i}.png'
        plt.savefig(filename, dpi=200)
        filenames.append(filename)

    plt.ioff()  # Turn off interactive mode
    plt.show()
    if make_gif:
        import imageio
        import os
        # Create a GIF
        with imageio.get_writer('my_animation.gif', mode='I', duration=0.1) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Remove files
        for filename in filenames:
            os.remove(filename)

    # plt.ioff()  # Turn off interactive mode
    # plt.show()

# filenames = []
# for i in range(num_time_steps):
#     ax.clear()  # Clear previous points
#     axis_equal(ax, simulator['eel'][:,:,:,0], simulator['eel'][:,:,:,1], simulator['eel'][:,:,:,2])
#     data = simulator['eel'][i]
#     x = data[:,:,0].flatten()
#     y = data[:,:,1].flatten()
#     z = data[:,:,2].flatten()

#     ax.scatter(x, y, z)
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')

#     velocity_data = simulator['eel_velocity'][i]
#     u = velocity_data[:, :, 0].flatten()  # Velocity components in x
#     v = velocity_data[:, :, 1].flatten()  # Velocity components in y
#     w = velocity_data[:, :, 2].flatten()  # Velocity components in z

#     # Scatter plot for the mesh points
#     ax.scatter(x, y, z, color='b')

#     # Adding velocity arrows
#     panel_center = (0.25*(simulator['eel'][i, 0:-1, 0:-1, :] + simulator['eel'][i, 0:-1, 1:, :] + simulator['eel'][i, 1:, 0:-1, :] + simulator['eel'][i, 1:, 1:, :])).reshape(-1, 3)

#     ax.quiver(panel_center[:,0], panel_center[:,1], panel_center[:, 2],
#                     u, v, w, color='r')
#     # change the view angle to x,y plane
#     # ax.view_init(90, -90)


#     plt.draw()
#     plt.pause(0.1)  # Pause to update the plot
#     time.sleep(0.01)  # Adjust as per your time step's actual duration


#     # Save the current frame
#     filename = f'frame_{i}.png'
#     plt.savefig(filename, dpi=200)
#     filenames.append(filename)

# import imageio
# import os
# # Create a GIF
# with imageio.get_writer('my_animation.gif', mode='I', duration=0.1) as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)

# # Remove files
# for filename in filenames:
#     os.remove(filename)

# plt.ioff()  # Turn off interactive mode
# plt.show()