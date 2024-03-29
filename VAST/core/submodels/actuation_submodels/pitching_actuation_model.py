import csdl
import numpy as np
from numpy.core.fromnumeric import size


class PitchingActuationModel(csdl.Model):
    """
    Compute the mesh at each time step for the eel actuation model given the kinematic variables.
    # the geometry and kinematics of the eel
    # is define from https://www.cse-lab.ethz.ch/wp-content/papercite-data/pdf/kern2006b.pdf

    Inputs
    ----------
    tail_amplitude : csdl variable [1,]
    tail_frequency : csdl variable [1,]
    wave_number : csdl variable [1,]
    linear_relation : csdl variable [1,]

    Parameters
    ----------
    surface_names : list
    surface_shapes : list
    n_period : int
    s_1_ind : int (num_pts for the head region)
    s_2_ind : int (num_pts for the tail region)

    Returns
    -------
    1. mesh[num_nodes,num_pts_chord, num_pts_span, 3] : csdl array
    bound vortices points
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('n_period')
        self.parameters.declare('s_1_ind',default=3)
        self.parameters.declare('s_2_ind',default=None)


    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        N_period = self.parameters['n_period']
        s_1_ind = self.parameters['s_1_ind'] # head region
        s_2_ind = self.parameters['s_2_ind'] # tail region
        if s_2_ind==None:
            s_2_ind = int(surface_shapes[0][1]-2)

        num_surface = len(surface_names)
        num_nodes = surface_shapes[0][0]

        ########################################################
        # 1. Defining kinematic variables for the fish
        ########################################################

        tail_amplitude = self.declare_variable('tail_amplitude')
        tail_frequency = self.declare_variable('tail_frequency')
        wave_number = self.declare_variable('wave_number')
        linear_relation = self.declare_variable('linear_relation')

        omg = 2*np.pi*tail_frequency
        
        t_temp_val = np.linspace(0, N_period, num_nodes)
        t_temp = self.create_input('t_temp', val = t_temp_val)

        t = t_temp/csdl.expand(tail_frequency,shape=t_temp.shape)
        ########################################################

        ########################################################
        # 2. Defining the mesh of the fish
        ########################################################

        for i in range(num_surface):
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]    

            L = 1.0 # length of the fish
            s1 = 0.04 * L # head region
            s2 = 0.95 * L # tail region  

            # define the x (length) discretization of the mesh:
            # cos spacing for head
            x_1 = (1-np.cos(np.linspace(0, np.pi/2,s_1_ind,endpoint=False)))*s1 
            # linear spacing for body and tail
            x_2 = np.linspace(s1, s2, int(s_2_ind-s_1_ind),endpoint=False) 
            x_3 = np.linspace(s2, L, int(nx-s_2_ind))
            x = np.concatenate((x_1,x_2,x_3)) # size = nx
                        
            # define the x and y coordinates of the mesh
            tensor_x = np.outer(x, np.ones(ny)).reshape(nx,ny,1) # x coordinate
            # for the z coordinate, tensor_z is just a expanded array from -0.5 to 0.5,
            # need to scale it to the actual height of the fish
            tensor_z  = np.outer(np.arange(ny), np.ones(nx)).T.reshape(nx,ny,1)/(ny-1) - 0.5 

            tensor_x_exp = np.einsum('i,jkl->ijkl',np.ones(num_nodes),tensor_x)
            tensor_z_exp = np.einsum('i,jkl->ijkl',np.ones(num_nodes),tensor_z)

            x_exp = self.declare_variable('mesh_x',val=tensor_x_exp)
            tensor_z_csdl = self.declare_variable('mesh_y',val=tensor_z_exp)

            a = 0.51*L
            b = 0.08*L 
            # from https://www.cse-lab.ethz.ch/wp-content/papercite-data/pdf/kern2006b.pdf to define the height of the fish
            # h(x) = b (1 - ((x-a)/a)^2 )^0.5
            height = np.einsum('ik,j->ijk',np.ones((num_nodes,ny)),b*(1-((x-a)/a)**2)**0.5).reshape(num_nodes,nx,ny,1)

            t_exp = csdl.expand(t, shape=(num_nodes,nx,ny,1),indices='i->ijkl')
            tail_amplitude_exp = csdl.expand(tail_amplitude, shape=(num_nodes,nx,ny,1),indices='i->ijkl')
            wave_number_exp = csdl.expand(wave_number, shape=(num_nodes,nx,ny,1),indices='i->ijkl')
            linear_relation_exp = csdl.expand(linear_relation, shape=(num_nodes,nx,ny,1),indices='i->ijkl') 
            omg_exp = csdl.expand(omg, shape=(num_nodes,nx,ny,1),indices='i->ijkl')
            
            # y = -A(x+0.03/1.03)*sin(2pix/lambda - omg*t)
            # ydot = -A(x+0.03/1.03)*cos(2pix/lambda - omg*t)*(-omg)

            # y is the lateral displacement of the fish
            # y = 0.125 * (s+0.03125)/1.03125 * sin(2*pi*x/lambda - omg*t)
            # ydot = 0.125 * (s+0.03125)/1.03125 * cos(2*pi*x/lambda - omg*t) * (-omg)

            # y = tail_amplitude_exp*((x_exp+linear_relation_exp)/(linear_relation_exp+1)) * csdl.sin(np.pi*2*x_exp/wave_number_exp - omg_exp*t_exp)
            # y_dot =  tail_amplitude_exp*((x_exp+linear_relation_exp)/(linear_relation_exp+1))*csdl.cos(np.pi*2*x_exp/wave_number_exp - omg_exp*t_exp)*(-omg_exp)

            y = tail_amplitude_exp*((x_exp+linear_relation_exp)/(linear_relation_exp+1)) * csdl.sin( - omg_exp*t_exp)
            y_dot =  tail_amplitude_exp*((x_exp+linear_relation_exp)/(linear_relation_exp+1))*csdl.cos( - omg_exp*t_exp)*(-omg_exp)

            # velocity of the fish on its collocation points
            coll_vel = self.create_output(name=surface_names[i]+'_coll_vel',val=np.zeros((num_nodes,nx-1,ny-1,3)))
            
            coll_vel[:,:,:,1] = (y_dot[:,:-1,:-1,:]+y_dot[:,1:,:-1,:]+y_dot[:,:-1,1:,:]+y_dot[:,1:,1:,:])/4 
            
            mesh = self.create_output(surface_names[i],val=np.zeros((num_nodes,nx,ny,3)))


            mesh[:,:,:,0] = x_exp    
            mesh[:,:,:,1] = y
            mesh[:,:,:,2] = tensor_z_csdl * height * 2 
            # this is due to the fact that the original height is from -0.5 to 0.5, need to scale it to the actual height of the fish


if __name__ == "__main__":
    import python_csdl_backend
    import numpy as np
    import pyvista as pv
    # simulator_name = 'csdl_om'
    num_nodes = 20
    num_pts_chord = 13 # nx
    num_pts_span = 3
    N_period=1
    # tail_amplitude = 0.125
    # tail_frequency = 0.48
    # vx = 0.38643524

    tail_amplitude = 0.06939378
    tail_frequency = 0.2
    vx = 0.38108754

    model = csdl.Model()
    # model.create_input('tail_amplitude',val=0.06939378)
    # model.create_input('tail_frequency',val=0.2)

    model.create_input('tail_amplitude',val=tail_amplitude)
    model.create_input('tail_frequency',val=tail_frequency)

    t_temp_val = np.linspace(0,N_period/0.48,num_nodes)

    model.add(EelActuationModel(surface_names=['surface'],surface_shapes=[(num_nodes,num_pts_chord,num_pts_span)],n_period=N_period),'actuation')
    

    sim = python_csdl_backend.Simulator(model)
    sim.run()

    surface = sim["surface"]
    print(surface.shape)

    for i in range(num_nodes):
        # x = surface[i,:,:,0] - t_temp_val[i]*0.38108754
        x = surface[i,:,:,0] - t_temp_val[i]*vx
        y = surface[i,:,:,1]
        z = surface[i,:,:,2]

        grid = pv.StructuredGrid(x,y,z)
        grid.save(filename=f'eel_actuation_{i}.vtk')
