import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix

body_order = {'pHipOrigin': 0,
              'jL5S1': 1,
              'jL4L3': 2,
              'jL1T12': 3,
              'jT9T8': 4,
              'jT1C7': 5,
              'jC1Head': 6,
              'jRightT4Shoulder': 7,
              'jRightShoulder': 8,
              'jRightElbow': 9,
              'jRightWrist': 10,
              'jLeftT4Shoulder': 11,
              'jLeftShoulder': 12,
              'jLeftElbow': 13,
              'jLeftWrist': 14,
              'jRightHip': 15,
              'jRightKnee': 16,
              'jRightAnkle': 17,
              'jRightBallFoot': 18,
              'jLeftHip': 19,
              'jLeftKnee': 20,
              'jLeftAnkle': 21,
              'jLeftBallFoot': 22}

ljt_order = {'jLeftWrist': 0, 'jLeftFirstCMC': 1, 'jLeftSecondCMC': 2, 'jLeftThirdCMC': 3, 'jLeftFourthCMC': 4, 'jLeftFifthCMC': 5, 'jLeftFifthMCP': 6, 'jLeftFifthPIP': 7, 'jLeftFifthDIP': 8, 'pLeftFifthTip': 9, 'jLeftFourthMCP': 10, 'jLeftFourthPIP': 11, 'jLeftFourthDIP': 12, 'pLeftFourthTip': 13, 'jLeftThirdMCP': 14, 'jLeftThirdPIP': 15, 'jLeftThirdDIP': 16, 'pLeftThirdTip': 17, 'jLeftSecondMCP': 18, 'jLeftSecondPIP': 19, 'jLeftSecondDIP': 20, 'pLeftSecondTip': 21, 'jLeftFirstMCP': 22, 'jLeftIP': 23, 'pLeftFirstTip': 24}
rjt_order = {'jRightWrist': 0, 'jRightFirstCMC': 1, 'jRightSecondCMC': 2, 'jRightThirdCMC': 3, 'jRightFourthCMC': 4, 'jRightFifthCMC': 5, 'jRightFifthMCP': 6, 'jRightFifthPIP': 7, 'jRightFifthDIP': 8, 'pRightFifthTip': 9, 'jRightFourthMCP': 10, 'jRightFourthPIP': 11, 'jRightFourthDIP': 12, 'pRightFourthTip': 13, 'jRightThirdMCP': 14, 'jRightThirdPIP': 15, 'jRightThirdDIP': 16, 'pRightThirdTip': 17, 'jRightSecondMCP': 18, 'jRightSecondPIP': 19, 'jRightSecondDIP': 20, 'pRightSecondTip': 21, 'jRightFirstMCP': 22, 'jRightIP': 23, 'pRightFirstTip': 24}
lp_order = {'jLeftWrist': 0, 'jLeftFirstCMC': 1, 'jLeftSecondCMC': 2, 'jLeftThirdCMC': 3, 'jLeftFourthCMC': 4, 'jLeftFifthCMC': 5, 'jLeftFifthMCP': 6, 'jLeftFifthPIP': 7, 'jLeftFifthDIP': 8, 'jLeftFourthMCP': 9, 'jLeftFourthPIP': 10, 'jLeftFourthDIP': 11, 'jLeftThirdMCP': 12, 'jLeftThirdPIP': 13, 'jLeftThirdDIP': 14, 'jLeftSecondMCP': 15, 'jLeftSecondPIP': 16, 'jLeftSecondDIP': 17, 'jLeftFirstMCP': 18, 'jLeftIP': 19}
rp_order = {'jRightWrist': 0, 'jRightFirstCMC': 1, 'jRightSecondCMC': 2, 'jRightThirdCMC': 3, 'jRightFourthCMC': 4, 'jRightFifthCMC': 5, 'jRightFifthMCP': 6, 'jRightFifthPIP': 7, 'jRightFifthDIP': 8, 'jRightFourthMCP': 9, 'jRightFourthPIP': 10, 'jRightFourthDIP': 11, 'jRightThirdMCP': 12, 'jRightThirdPIP': 13, 'jRightThirdDIP': 14, 'jRightSecondMCP': 15, 'jRightSecondPIP': 16, 'jRightSecondDIP': 17, 'jRightFirstMCP': 18, 'jRightIP': 19}

def makeTpose(bone_vector):
    bodyTpose, handTpose = torch.zeros(22,3), torch.zeros(2,24,3)
    for p_nm, child_lst in bone_vector["body"].items():
        for c_nm, bone in child_lst.items():
            if c_nm[-3:] != "Toe":
                jtidx = body_order[c_nm]-1
                bodyTpose[jtidx] = torch.tensor(bone, requires_grad=False)


    for p_nm, child_lst in bone_vector["lhand"].items():
        for c_nm, bone in child_lst.items():
            if c_nm[0] != "p" or c_nm[-3:] == "Tip":
                jtidx = ljt_order[c_nm]-1
                handTpose[0][jtidx] = torch.tensor(bone, requires_grad=False)


    for p_nm, child_lst in bone_vector["rhand"].items():
        for c_nm, bone in child_lst.items():
            if c_nm[0] != "p" or c_nm[-3:] == "Tip":
                jtidx = (rjt_order[c_nm]-1)
                handTpose[1][jtidx] = torch.tensor(bone, requires_grad=False)
    return bodyTpose.reshape(-1), handTpose.reshape(2, 72)


def makeHandMapper():
    base_array = torch.zeros((20,72,72), dtype=torch.float32) # concat
    base_array[0, :15, :15] = torch.eye(15)
    # fi+1 : finger number
    for fi in range(5):
        fi_idx = 4-fi
        if fi == 0:
            # Rwrist 
            row_start, row_end = 15+12*fi_idx, 72
            col_start = 15+12*fi_idx # 63
            # wrist rot 
            base_array[0, 15+12*fi_idx:, :3] = torch.cat([torch.eye(3) for _ in range(3)], dim=0)
            # CMC 전용
            base_array[fi+1, row_start:row_end, col_start:col_start+3] = torch.cat([torch.eye(3) for _ in range(3)], dim=0) # 4*fi_idx+1
            base_array[3*(fi_idx+2), row_start+3:row_end, col_start+3:col_start+6] = torch.cat([torch.eye(3) for _ in range(2)], dim=0)
            base_array[3*(fi_idx+2)+1, row_start+6:row_end, col_start+6:col_start+9] = torch.eye(3)
        else:
            row_start, row_end = 15+12*fi_idx, 15+12*(fi_idx+1)
            col_start = 15+12*fi_idx
            # wrist rot 
            base_array[0, row_start:row_end, 3*fi:3*(fi+1)] = torch.cat([torch.eye(3) for _ in range(4)], dim=0)
            # Rot CMC 전용
            base_array[fi+1, row_start:row_end, col_start:col_start+3] = torch.cat([torch.eye(3) for _ in range(4)], dim=0) # fi == 4
            base_array[3*(fi_idx+2), row_start+3:row_end, col_start+3:col_start+6] = torch.cat([torch.eye(3) for _ in range(3)], dim=0)
            base_array[3*(fi_idx+2)+1, row_start+6:row_end, col_start+6:col_start+9] = torch.cat([torch.eye(3) for _ in range(2)], dim=0)
            base_array[3*(fi_idx+2)+2, row_start+9:row_end, col_start+9:col_start+12] = torch.eye(3)
    base_array = torch.cat([base_array[None,:], base_array[None,:]], dim=0)
    return base_array # 2,20,72,72

class HandMaker(nn.Module):
    def __init__(self, bone_length):
        super().__init__()
        """
        bone_length : (48,6 => 2, 72)
        """
        mapped_reltrans = torch.einsum('SJMK, SK -> SJM',makeHandMapper(), bone_length)  # mapped bonelength
        mapped_reltrans = mapped_reltrans.reshape(2,20,24,3)
        self.mapper = nn.Parameter(mapped_reltrans, requires_grad=False)

    def forward(self, orientation):
        """
        Orientation : torch.tensor (F, 40,6) rotation value 
        """
        num_frame = orientation.shape[0]
        wrist_pos = torch.zeros((num_frame, 2,1,3)).to(orientation.device)
        mat = rotation_6d_to_matrix(orientation) # (40,3,3) z xc
        mat = mat.view(num_frame, 2,20,3,3)

        rel_mapped = torch.einsum('FSRMK, SRJK -> FSRJM', mat, self.mapper) # 
        acq_joints = torch.sum(rel_mapped, dim=2) # (2,24,3)
        out = torch.cat([wrist_pos, acq_joints], dim=2)
        return out


def makeBodyMapper():
    base_array = torch.zeros((23,23*3, 22*3), dtype=torch.float32) # concat
    for i in range(23):
        if i == 0:
            base_array[i][1*3:15*3 ,:3] =  torch.cat([torch.eye(3) for _ in range(14)], axis=0)
            base_array[i][15*3:19*3 ,14*3:15*3] = torch.cat([torch.eye(3) for _ in range(4)], axis=0)
            base_array[i][19*3:23*3 ,18*3:19*3] = torch.cat([torch.eye(3) for _ in range(4)], axis=0)
        elif (i >= 1 and i <= 3):
            start_row = i+1
            end_row = 15
            base_array[i][start_row*3:end_row*3, 3*i:3*(i+1)] = torch.cat([torch.eye(3) for _ in range(14-i)], axis=0)
        elif i == 4: # jT9T8
            start_row = i+1
            end_row = 7
            base_array[i][start_row*3:end_row*3, 3*i:3*(i+1)] = torch.cat([torch.eye(3) for _ in range(2)], axis=0)
            # shoulders
            base_array[i][end_row*3:(end_row+4)*3, 3*(i+2):3*(i+3)] = torch.cat([torch.eye(3) for _ in range(4)], axis=0) # RT4shoulder
            base_array[i][(end_row+4)*3:(end_row+8)*3, 3*(i+6):3*(i+7)] = torch.cat([torch.eye(3) for _ in range(4)], axis=0) # LT4shoulder
        elif i == 5:
            start_row = i+1
            end_row = 7
            base_array[i][start_row*3:end_row*3, 3*i:3*(i+1)] = torch.cat([torch.eye(3) for _ in range(1)], axis=0)
        elif i >= 7  :
            start_col, start_row = i*3, (i+1)*3
            num_ident = 3 - (i-7) % 4
            end_row = start_row + 3*num_ident
            if num_ident >= 1:
                base_array[i][start_row:end_row, start_col:start_col+3] = torch.cat([torch.eye(3) for _ in range(num_ident)], axis=0) # LT4shoulder
    return base_array

class BodyMaker(nn.Module):
    def __init__(self, bone_trans):
        super().__init__()
        """
        bonetrans : (22, 3) => (66,)
        """
        mapped_reltrans = torch.einsum('JMN, N -> JM', makeBodyMapper(), bone_trans)
        mapped_reltrans = mapped_reltrans.reshape(23, 23, 3)
        self.mapper = nn.Parameter(mapped_reltrans, requires_grad=False)

    def forward(self, orientation):
        """
        Orientation : torch.tensor (F,23,6) rotation value 
        """
        mat = rotation_6d_to_matrix(orientation)
        batched = torch.einsum('FJMN, JLN -> FJLM', mat, self.mapper)

        return torch.sum(batched, dim=1)#, batched # (23,3)



def getRotation(vec1, vec2):
    vec1 = vec1/np.sqrt(vec1[0]**2+vec1[1]**2+vec1[2]**2)
    vec2 = vec2/np.sqrt(vec2[0]**2+vec2[1]**2+vec2[2]**2)

    n = np.cross(vec1, vec2)

    v_s = np.sqrt(n[0]**2+n[1]**2+n[2]**2)
    v_c = np.dot(vec1, vec2)
    skew = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    rotmat = np.eye(3) + skew+ skew@skew*((1-v_c)/(v_s**2))
    return rotmat


def get_stickhand(hand_arr, conn=None, color=[0.4, 0.4, 0.4]):
    conn = [[0, 1], [0, 2], [0, 3], [0, 4],[0, 5], [5, 6], [6, 7], [7, 8], [8, 9],
            [4, 10], [10, 11], [11, 12], [12, 13], [3, 14], [14, 15], [15, 16], [16, 17],
            [2, 18], [18, 19], [19, 20], [20, 21], [1, 22], [22, 23], [23, 24]]
    def get_sphere(position, radius, color):
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=radius).paint_uniform_color(color)
        sp.translate(position)
        sp.compute_vertex_normals()
        return sp
    def get_segment(parent, child, radius, color):
        v = parent-child
        seg = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(v), resolution=20, split=1).paint_uniform_color(color)
        mat = getRotation(vec1=np.array([0, 0, 1]), vec2=v/np.linalg.norm(v))
        seg.rotate(mat)
        seg.translate((parent+child)/2)
        seg.compute_vertex_normals()
        return seg
    mesh = o3d.geometry.TriangleMesh()
    for i in range(hand_arr.shape[0]):
        mesh += get_sphere(hand_arr[i], 0.003, [0.5,0,0.5])
    for pairs in conn:
        p, c = pairs[0], pairs[1]
        mesh += get_segment(hand_arr[p], hand_arr[c], 0.005, color)
    return mesh


def get_stickman(body_position_arr, head_tip=None, color=[0.4, 0.4, 0.4], foot_contact=None):
    body_line_idxs_int = [[0, 1], [0, 15], [0, 19], [1, 2], [2, 4], [4, 5], [4, 7], [4, 11], [5, 6],
                          [7, 8], [8, 9], [9, 10], [11, 12], [12, 13], [13, 14], [15, 16], [16, 17],
                          [17, 18], [19, 20], [20, 21], [21, 22]]
    def get_sphere(position, radius, color):
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=radius).paint_uniform_color([1,0,1])  # in rgb 0~1
        sp.translate(position)
        sp.compute_vertex_normals()
        return sp
    def get_segment(parent, child, radius, color):
        v = parent-child
        seg = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(v), resolution=20, split=1).paint_uniform_color(color)
        mat = getRotation(vec1=np.array([0, 0, 1]), vec2=v/np.linalg.norm(v))
        seg.rotate(mat)
        seg.translate((parent+child)/2)
        seg.compute_vertex_normals()
        return seg
    def get_foot(parent, child, color, width=0.05):
        v = parent-child
        height = np.linalg.norm(v)
        depth = width/2
        mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth).paint_uniform_color(color)
        mesh.translate([-width/2, -height/2, -depth/2])
        mat = getRotation(vec1=np.array([0, 1, 0]), vec2=v/height)
        mesh.rotate(mat)
        mesh.translate((parent+child)/2)
        mesh.compute_vertex_normals()
        return mesh
    body_mesh = o3d.geometry.TriangleMesh()
    for bidx in range(23):
        if bidx == 6 and head_tip is not None: # jC1Head
            body_mesh += get_segment(body_position_arr[6], head_tip, 0.08, color)
        elif bidx in [12,8,15,19]: # 'jLeftShoulder','jRightShoulder','jRightHip','jLeftHip'
            body_mesh += get_sphere(body_position_arr[bidx], 0.03, color)
        elif bidx == 22:#"jLeftBallFoot":
            if foot_contact is not None:
                if foot_contact[22]:
                    body_mesh += get_sphere(body_position_arr[bidx], 0.03, [1,0,0])
                else:
                    body_mesh += get_sphere(body_position_arr[bidx], 0.03, color)

        elif bidx == 18:#"jRightBallFoot":
            if foot_contact is not None:
                if foot_contact[18]:
                    body_mesh += get_sphere(body_position_arr[bidx], 0.03, [1,0,0])
                else:
                    body_mesh += get_sphere(body_position_arr[bidx], 0.03, color)

        else:
            body_mesh += get_sphere(body_position_arr[bidx], 0.02, color)

    for pidx, cidx in body_line_idxs_int:
        parent, child = body_position_arr[pidx], body_position_arr[cidx]
        if pidx == 21 or pidx == 17:
            body_mesh += get_foot(parent, child, width=0.05, color=color)
        else:
            body_mesh += get_segment(parent, child, 0.02, color)
    return body_mesh

class simpleViewerHeadless(object):
    
    
    def __init__(self, title, width, height, view_set_list, view=None):
        mesh_default_material = o3d.visualization.rendering.MaterialRecord()
        mesh_default_material.shader = "defaultLit"
        
        line_default_material = o3d.visualization.rendering.MaterialRecord()
        line_default_material.shader = "unlitLine"
        
        self.default_materials = {
            "mesh": mesh_default_material,
            "line": line_default_material,
        }
        self.main_vis = o3d.visualization.rendering.OffscreenRenderer(width, height)
        
        if view is not None:
            self.intrinsic = view.intrinsic.intrinsic_matrix
        self.width = width
        self.height = height


    def setupcamera(self, extrinsic_matrix):
        self.main_vis.setup_camera(self.intrinsic, extrinsic_matrix, self.width, self.height)

    def grab_render(self, *keys):
        render = dict()
        for key in keys:
            if key == "rgb":
                image = np.asarray(self.main_vis.render_to_image())
            elif key == "depth":
                image = np.asarray(self.main_vis.render_to_depth_image(z_in_view_space=True))
            else:
                raise ValueError(f"Unknown render key: {key}")
            render[key] = image
        return render

    def add_plane(self, resolution=128, bound=100, up_vec="z"):
        """
        Creates a plane as a subdivided TriangleMesh (instead of a wireframe LineSet).
        This plane has 'resolution x resolution' vertices spread from -bound to +bound,
        and each cell is turned into two triangles.

        :param resolution: Number of vertices along each axis (>= 2).
        :param bound: Half-extent of the plane along each axis, so plane goes [-bound, +bound].
        :param up_vec: Which axis is "up"; can be 'z', 'y', or 'x'.
        """

        def make_filled_plane(bound=100.0, resolution=128, up="z"):
            # 1) Create a grid of coordinates in 2D (ranging from -bound to +bound).
            xs = np.linspace(-bound, bound, num=resolution, dtype=np.float32)
            ys = np.linspace(-bound, bound, num=resolution, dtype=np.float32)

            # We’ll generate resolution x resolution vertices in 3D.
            # The shape is (resolution*resolution, 3).
            # We adapt their final positions based on the 'up' axis.
            verts_2d = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(
                -1, 2
            )  # shape: [N, 2]

            if up == "z":
                # XY plane, Z=0
                verts_3d = np.column_stack(
                    (verts_2d[:, 0], verts_2d[:, 1], np.zeros_like(verts_2d[:, 0]))
                )
            elif up == "y":
                # XZ plane, Y=0
                verts_3d = np.column_stack(
                    (verts_2d[:, 0], np.zeros_like(verts_2d[:, 0]), verts_2d[:, 1])
                )
            elif up == "x":
                # YZ plane, X=0
                verts_3d = np.column_stack(
                    (np.zeros_like(verts_2d[:, 0]), verts_2d[:, 0], verts_2d[:, 1])
                )
            else:
                raise ValueError("up_vec must be 'x', 'y', or 'z'")

            # 2) Create the triangle indices.
            # For each cell in the (resolution-1) x (resolution-1) grid, we make two triangles.
            triangles = []
            for row in range(resolution - 1):
                for col in range(resolution - 1):
                    i0 = row * resolution + col
                    i1 = i0 + 1
                    i2 = (row + 1) * resolution + col
                    i3 = i2 + 1
                    # Two triangles: (i0, i2, i1) and (i2, i3, i1)
                    triangles.append([i0, i2, i1])
                    triangles.append([i2, i3, i1])

            # 3) Build a TriangleMesh from these vertices and triangles.
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts_3d)
            mesh.triangles = o3d.utility.Vector3iVector(
                np.array(triangles, dtype=np.int32)
            )
            mesh.compute_triangle_normals()

            # Optional: paint uniform color (vertex colors).
            # This is only needed if you plan on using "defaultUnlit" or "defaultLit" with vertex colors.
            # If you have a separate MaterialRecord with a base color, you can skip this.
            mesh.paint_uniform_color([0.5, 0.5, 0.5])

            return mesh

        # Create the plane mesh:
        plane_mesh = make_filled_plane(bound, resolution, up=up_vec)

        # Add it to your scene with a default or custom material:
        self.add_geometry({"name": "floor", "geometry": plane_mesh})

        return

    def remove_plane(self):
        self.main_vis.scene.remove_geometry("floor")
        return

    def add_geometry(self, geometry:dict):
        material = None
        if "LineSet" in str(type(geometry["geometry"])):
            material = self.default_materials["line"]
        elif "TriangleMesh" in str(type(geometry["geometry"])):
            material = self.default_materials["mesh"]
        self.main_vis.scene.add_geometry(**geometry, material=material)


    def transform(self,name, transform_mtx):
        self.main_vis.scene.set_geometry_transform(name, transform_mtx)

    def set_background(self, image):
        self.main_vis.scene.set_background([1, 1, 1, 0], image)

    def remove_geometry(self, geom_name):
        self.main_vis.scene.remove_geometry(geom_name)

if __name__ == '__main__':
    x = simpleViewerHeadless("Render Scene", 1600, 800, [], view)  #

    vis = o3d.visualization.rendering.OffscreenRenderer(1600, 800)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"  # or "defaultUnlit", etc.

    vis.scene.add_geometry("global", global_coord, material)

    material_2 = o3d.visualization.rendering.MaterialRecord()
    material_2.shader = "unlitLine"

    vis.scene.add_geometry(
        name="floor",
        geometry=plane,  # This is a LineSet
        material=material_2,
    )

    vis.scene.add_geometry(
        **{"name": "human", "geometry": bmesh + hmesh, "material": material}
    )
    vis.scene.reset_camera_to_default()
    view.intrinsic.intrinsic_matrix
    vis.setup_camera(camera_matrix, extrinsic_matrix, width, height)
    img = vis.render_to_image()  # Renders the scene from your specified camera setup.

    from matplotlib import pyplot as plt

    plt.imshow(img)
    plt.show()

    str(type(plane))
