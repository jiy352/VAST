from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show
import numpy as np

def make_video(surface_properties_dict,num_nodes, sim, xrange=(0, 35), yrange=(-10, 10), zrange=(-3, 0.5)):    
    surface_names = list(surface_properties_dict.keys())
    nt = num_nodes - 1 
    
    axs = Axes(
        xrange=xrange,
        yrange=yrange,
        zrange=zrange,
    )
    video = Video("spider.gif", duration=10, backend='ffmpeg')
    for i in range(nt - 1):
        vp = Plotter(
            bg='beige',
            bg2='lb',
            # axes=0,
            #  pos=(0, 0),
            offscreen=False,
            interactive=1)
        # Any rendering loop goes here, e.g.:
        for surface_name in surface_names:
            vps = Points(np.reshape(sim[surface_name][i, :, :, :], (-1, 3)),
                        r=8,
                        c='red')
            vp += vps
            vp += __doc__
            vps = Points(np.reshape(sim['op_'+surface_name+'_wake_coords'][i, 0:i, :, :],
                                    (-1, 3)),
                        r=8,
                        c='blue')
            vp += vps
            vp += __doc__
        vp.show(axs, elevation=-60, azimuth=-0,
                axes=False)  # render the scene
        video.addFrame()  # add individual frame
        # time.sleep(0.1)
        # vp.interactive().close()
        vp.closeWindow()
    vp.closeWindow()
    video.close()  # merge all the recorded frames

