def get_mark_direction(a, b):
        return (a * a) / (a * a + b * b)

def init_direction(smplx_para):
    import numpy as np
    root_pose = smplx_para['smplx_root_pose'][0]
    from scipy.spatial.transform import Rotation
    r = Rotation.from_rotvec(root_pose)
    rotation_matrix = r.as_matrix()
    vector_direction = np.array([0, 0, 1])
    vector_direction = np.dot(rotation_matrix, vector_direction)
    
    if vector_direction[2] < 0 and np.abs(vector_direction[2]) > np.abs(vector_direction[0]):
        direction = "front"
    elif vector_direction[2] > 0 and np.abs(vector_direction[2]) > np.abs(vector_direction[0]):
        direction = "back"
    elif vector_direction[0] < 0 and np.abs(vector_direction[0]) > np.abs(vector_direction[2]):
        direction = "left"
    elif vector_direction[0] > 0 and np.abs(vector_direction[0]) > np.abs(vector_direction[2]):
        direction = "right"
    
    if direction in ['front', 'back']:
        mark_direction = str(get_mark_direction(vector_direction[2], vector_direction[0]))
    elif direction in ['left', 'right']:
        mark_direction = str(get_mark_direction(vector_direction[0], vector_direction[2]))
    else:
        mark_direction = str(0)

    vector_direction = [str(item) for item in vector_direction]
    return direction, vector_direction, mark_direction