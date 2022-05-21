from xml.dom import minidom
import os

class CreateRobotSDF():
    def __init__(self) -> None:
        self.root = minidom.Document()
        self.sdf_tag = self.root.createElement('sdf')
        self.sdf_tag.setAttribute('version', '1.5')
        self.root.appendChild(self.sdf_tag)
        self.model_tag = self.root.createElement('model')
        self.model_tag.setAttribute('name', 'stewart')
        self.sdf_tag.appendChild(self.model_tag)


    def add_plugin(self, name, filename):
        plugin_tag = self.root.createElement("plugin")
        plugin_tag.setAttribute('name',name)
        plugin_tag.setAttribute('filename',filename)
        self.model_tag.appendChild(plugin_tag)


    def add_joint(self,name, type, parent, child,pose , axis_xyz, axis_limit_lower_param=False,axis_limit_upper_param=False, axis_limit_velocity_param=False, axis_limit_effort_param=False, axis_dynamics_damping_param=False, axis_name="axis", axis_name_2=False, axis_name_2_xyz=False):
        joint_tag = self.root.createElement("joint")
        joint_tag.setAttribute('name',name)
        joint_tag.setAttribute('type',type)
        self.model_tag.appendChild(joint_tag)

        parent_tag = self.root.createElement("parent")
        parent_text_node = self.root.createTextNode(parent) 
        joint_tag.appendChild(parent_tag)
        parent_tag.appendChild(parent_text_node)

        child_tag = self.root.createElement("child")
        child_text_node = self.root.createTextNode(child) 
        joint_tag.appendChild(child_tag)
        child_tag.appendChild(child_text_node)


        pose_tag = self.root.createElement("pose")
        pose_text_node = self.root.createTextNode(pose) 
        joint_tag.appendChild(pose_tag)
        pose_tag.appendChild(pose_text_node)

        self._add_axis(xyz_param=axis_xyz, limit_lower_param = axis_limit_lower_param, limit_upper_param=axis_limit_upper_param, limit_velocity_param=axis_limit_velocity_param,limit_effort_param=axis_limit_effort_param, dynamics_damping_param = axis_dynamics_damping_param, axis_name=axis_name )
        joint_tag.appendChild(self.axis_tag)

        if axis_name_2 and axis_name_2_xyz:
            self._add_axis(xyz_param=axis_name_2_xyz, limit_lower_param = axis_limit_lower_param, limit_upper_param=axis_limit_upper_param, limit_velocity_param=axis_limit_velocity_param,limit_effort_param=axis_limit_effort_param, dynamics_damping_param = axis_dynamics_damping_param, axis_name=axis_name_2 )
            joint_tag.appendChild(self.axis_tag)



    def _add_axis(self, xyz_param, limit_lower_param=False,limit_upper_param=False,limit_velocity_param=False,limit_effort_param=False,dynamics_damping_param=False, axis_name = "axis"):
        #create axis with defult name
        self.axis_tag = self.root.createElement(axis_name)

        # create axis main elements
        xyz_tag = self.root.createElement("xyz")
        limit_tag = self.root.createElement("limit")
        dynamics_tag = self.root.createElement("dynamics")

        # xyz related tag definition
        xyz_text_node = self.root.createTextNode(xyz_param) 
        xyz_tag.appendChild(xyz_text_node)

        self.axis_tag.appendChild(xyz_tag)
        self.axis_tag.appendChild(limit_tag)
        self.axis_tag.appendChild(dynamics_tag)

        # limit related tags definition
        limit_lower_tag = self.root.createElement("lower")
        if limit_lower_param:
            limit_lower_node = self.root.createTextNode(limit_lower_param) 
            limit_lower_tag.appendChild(limit_lower_node)
        limit_tag.appendChild(limit_lower_tag)
        
        limit_upper_tag = self.root.createElement("upper")
        if limit_upper_param:
            limit_upper_node = self.root.createTextNode(limit_upper_param) 
            limit_upper_tag.appendChild(limit_upper_node)
        limit_tag.appendChild(limit_upper_tag)

        limit_velocity_tag = self.root.createElement("velocity")
        if limit_velocity_param:
            limit_velocity_node = self.root.createTextNode(limit_velocity_param) 
            limit_velocity_tag.appendChild(limit_velocity_node)
        limit_tag.appendChild(limit_velocity_tag)


        limit_effort_tag = self.root.createElement("effort")
        if limit_effort_param:
            limit_effort_node = self.root.createTextNode(limit_effort_param) 
            limit_effort_tag.appendChild(limit_effort_node)
        limit_tag.appendChild(limit_effort_tag)

        # dynamics related tags definition
        dynamics_damping_tag = self.root.createElement("damping")
        if dynamics_damping_param:
            dynamics_damping_node = self.root.createTextNode(dynamics_damping_param) 
            dynamics_damping_tag.appendChild(dynamics_damping_node)
        dynamics_tag.appendChild(dynamics_damping_tag)


    def add_link(self, name, pose, geometry,mass, radius, length,material_script_uri_param,material_script_name_param):
        link_tag = self.root.createElement("link")
        link_tag.setAttribute('name',name)
        self.model_tag.appendChild(link_tag)  

        pose_tag = self.root.createElement("pose")
        pose_text_node = self.root.createTextNode(pose) 
        pose_tag.appendChild(pose_text_node)
        link_tag.appendChild(pose_tag)

        self._add_inertial(geometry=geometry,mass=mass, radius=radius, length=length)
        link_tag.appendChild(self.inertial_tag)


        self._add_visual(visual_name= name+"_visual", geometry=geometry,radius=radius, length=length, material_script_uri_param=material_script_uri_param,material_script_name_param=material_script_name_param)
        link_tag.appendChild(self.visual_tag)


        self._add_collision(name+"_collision", geometry=geometry,radius=radius, length=length)
        link_tag.appendChild(self.collistion_tag)



    def _add_inertial(self, geometry, mass, radius, length,whd=[1,1,1]):
        assert geometry in ['cylinder','sphere','box'],"geometry in not in ['cylinder_inertial','sphere_inertial','box_inertial']"
        
        self.inertial_tag = self.root.createElement("inertial")
        mass_tag = self.root.createElement("mass")
        mass_text_node = self.root.createTextNode(str(mass)) 
        mass_tag.appendChild(mass_text_node)  
        self.inertial_tag.appendChild(mass_tag)


        inertia_tag = self.root.createElement("inertia")

        if geometry =='cylinder':
            ixx_tag = self.root.createElement("ixx")
            ixx_text_node = self.root.createTextNode(str(1/12*mass*(3*radius**2 + length**2))) 
            ixx_tag.appendChild(ixx_text_node)
            inertia_tag.appendChild(ixx_tag)

            ixy_tag = self.root.createElement("ixy")
            ixy_text_node = self.root.createTextNode("0") 
            ixy_tag.appendChild(ixy_text_node)
            inertia_tag.appendChild(ixy_tag)

            ixz_tag = self.root.createElement("ixz")
            ixz_text_node = self.root.createTextNode("0") 
            ixz_tag.appendChild(ixz_text_node)
            inertia_tag.appendChild(ixz_tag)

            iyy_tag = self.root.createElement("iyy")
            iyy_text_node = self.root.createTextNode(str(1/12*mass*(3*radius**2 + length**2))) 
            iyy_tag.appendChild(iyy_text_node)
            inertia_tag.appendChild(iyy_tag)

            iyz_tag = self.root.createElement("iyz")
            iyz_text_node = self.root.createTextNode("0") 
            iyz_tag.appendChild(iyz_text_node)
            inertia_tag.appendChild(iyz_tag)

            izz_tag = self.root.createElement("izz")
            izz_text_node = self.root.createTextNode(str(1/2*mass*radius**2)) 
            izz_tag.appendChild(izz_text_node)
            inertia_tag.appendChild(izz_tag)

            # add inertia to inertial
            self.inertial_tag.appendChild(inertia_tag)

        if geometry =='sphere':
            ixx_tag = self.root.createElement("ixx")
            ixx_text_node = self.root.createTextNode(str(0.4*mass*radius**2 )) 
            ixx_tag.appendChild(ixx_text_node)
            inertia_tag.appendChild(ixx_tag)

            ixy_tag = self.root.createElement("ixy")
            ixy_text_node = self.root.createTextNode("0") 
            ixy_tag.appendChild(ixy_text_node)
            inertia_tag.appendChild(ixy_tag)

            ixz_tag = self.root.createElement("ixz")
            ixz_text_node = self.root.createTextNode("0") 
            ixz_tag.appendChild(ixz_text_node)
            inertia_tag.appendChild(ixz_tag)

            iyy_tag = self.root.createElement("iyy")
            iyy_text_node = self.root.createTextNode(str(0.4*mass*radius**2 ))    
            iyy_tag.appendChild(iyy_text_node)
            inertia_tag.appendChild(iyy_tag)

            iyz_tag = self.root.createElement("iyz")
            iyz_text_node = self.root.createTextNode("0") 
            iyz_tag.appendChild(iyz_text_node)
            inertia_tag.appendChild(iyz_tag)

            izz_tag = self.root.createElement("izz")
            izz_text_node = self.root.createTextNode(str(0.4*mass*radius**2 )) 
            izz_tag.appendChild(izz_text_node)
            inertia_tag.appendChild(izz_tag)

            # add inertia to inertial
            self.inertial_tag.appendChild(inertia_tag)


        if geometry =='box':
            w, h , d = whd # unpack whd list to weight , height and depth
            ixx_tag = self.root.createElement("ixx")
            ixx_text_node = self.root.createTextNode(str(1/12*mass*(h**2 + d**2 ))) 
            ixx_tag.appendChild(ixx_text_node)
            inertia_tag.appendChild(ixx_tag)

            ixy_tag = self.root.createElement("ixy")
            ixy_text_node = self.root.createTextNode("0") 
            ixy_tag.appendChild(ixy_text_node)
            inertia_tag.appendChild(ixy_tag)

            ixz_tag = self.root.createElement("ixz")
            ixz_text_node = self.root.createTextNode("0") 
            ixz_tag.appendChild(ixz_text_node)
            inertia_tag.appendChild(ixz_tag)

            iyy_tag = self.root.createElement("iyy")
            iyy_text_node = self.root.createTextNode(str(1/12*mass*(w**2 + d**2 )))    
            iyy_tag.appendChild(iyy_text_node)
            inertia_tag.appendChild(iyy_tag)

            iyz_tag = self.root.createElement("iyz")
            iyz_text_node = self.root.createTextNode("0") 
            iyz_tag.appendChild(iyz_text_node)
            inertia_tag.appendChild(iyz_tag)

            izz_tag = self.root.createElement("izz")
            izz_text_node = self.root.createTextNode(str(1/12*mass*(w**2 + h**2 ))) 
            izz_tag.appendChild(izz_text_node)
            inertia_tag.appendChild(izz_tag)

            # add inertia to inertial
            self.inertial_tag.appendChild(inertia_tag)


    def _add_visual(self, visual_name,geometry,radius,length,material_script_uri_param,material_script_name_param):
        #visual related tag
        self.visual_tag = self.root.createElement("visual")
        self.visual_tag.setAttribute('name',visual_name)

        geometry_tag = self.root.createElement("geometry")
        if geometry == 'cylinder':
            geom_tag = self.root.createElement("cylinder")
            radius_tag = self.root.createElement("radius")
            radius_text_node = self.root.createTextNode(str(radius))
            radius_tag.appendChild(radius_text_node)
            length_tag = self.root.createElement("length")
            length_text_node = self.root.createTextNode(str(length))
            length_tag.appendChild(length_text_node)
            geom_tag.appendChild(radius_tag)
            geom_tag.appendChild(length_tag)

            # add geom_tag to geometry
            geometry_tag.appendChild(geom_tag)
        elif geometry == 'sphere':
            geom_tag = self.root.createElement("sphere")
            radius_tag = self.root.createElement("radius")
            radius_text_node = self.root.createTextNode(str(radius))
            radius_tag.appendChild(radius_text_node)
            geom_tag.appendChild(radius_tag)

            # add geom_tag to geometry
            geometry_tag.appendChild(geom_tag)
           
        elif geometry == 'box':
            geom_tag = self.root.createElement("box")

        self.visual_tag.appendChild(geometry_tag)


        # material tag
        material_tag = self.root.createElement("material")  
        material_script_tag = self.root.createElement("script")
        
        material_script_uri_tag = self.root.createElement("uri")
        material_script_uri_node = self.root.createTextNode(material_script_uri_param)
        material_script_uri_tag.appendChild(material_script_uri_node) 
        material_script_tag.appendChild(material_script_uri_tag) 

        material_script_name_tag = self.root.createElement("name")
        material_script_name_node = self.root.createTextNode(material_script_name_param)
        material_script_name_tag.appendChild(material_script_name_node) 
        material_script_tag.appendChild(material_script_name_tag) 

        material_tag.appendChild(material_script_tag)

        # add material tag to visual
        self.visual_tag.appendChild(material_tag)


    def _add_collision(self, collision_name,geometry,radius,length):
        #collistion related tag
        self.collistion_tag = self.root.createElement("collision")
        self.collistion_tag.setAttribute('name',collision_name)

        geometry_tag = self.root.createElement("geometry")
        if geometry == 'cylinder':
            geom_tag = self.root.createElement("cylinder")
            radius_tag = self.root.createElement("radius")
            radius_text_node = self.root.createTextNode(str(radius))
            radius_tag.appendChild(radius_text_node)
            length_tag = self.root.createElement("length")
            length_text_node = self.root.createTextNode(str(length))
            length_tag.appendChild(length_text_node)
            geom_tag.appendChild(radius_tag)
            geom_tag.appendChild(length_tag)

            # add geometry tag to visual 
            geometry_tag.appendChild(geom_tag)
            
        elif geometry == 'sphere':
            geom_tag = self.root.createElement("sphere")
            radius_tag = self.root.createElement("radius")
            radius_text_node = self.root.createTextNode(str(radius))
            radius_tag.appendChild(radius_text_node)
            geom_tag.appendChild(radius_tag)

            # add geom_tag to geometry
            geometry_tag.appendChild(geom_tag)

        elif geometry == 'box':
            geom_tag = self.root.createElement("box")

        self.collistion_tag.appendChild(geometry_tag)



    def save_model(self, file_name):
        xml_str = self.root.toprettyxml(indent ="\t")
        save_path_file =  os.path.dirname(os.path.abspath(__file__))  + f"/{file_name}.sdf"
        with open(save_path_file, "w") as f:
            f.write(xml_str)

