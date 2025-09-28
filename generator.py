import bpy
import math
import random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from mathutils import Vector, Euler

class PalletBoxGenerator:
    def __init__(self):
        self.textures_dir = Path("./box_counter/objects")
        self.output_dir   = Path("./box_counter/box_stack")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.pallet_dims = Vector((0.8, 1.2, 0.145))
        self.pallet_pos  = Vector((0, 0, -self.pallet_dims.z/2))
        
        self.box_dims = {
            'group_box': Vector((0.411, 0.343, 0.284)),
            'laptop':    Vector((0.4,   0.27,  0.064)),
            'tablet':    Vector((0.29,  0.185, 0.05))
        }
        
        self.cameras = {
            'left':  Vector((-2.5,  2.5, 1.2)),
            'right': Vector(( 2.5, -2.5, 1.2))
        }
        
        self.max_stack_height = 4 * self.box_dims['group_box'].z

    def setup_gpu_cycles(self):
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.refresh_devices()
        for device in prefs.devices:
            if device.type in {'CUDA', 'OPTIX', 'HIP', 'METAL'}:
                device.use = True
        scene.cycles.device = 'GPU'
        scene.cycles.samples = 128
        scene.cycles.use_denoising = True

    def clear_scene(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                          bpy.data.lights, bpy.data.cameras):
            for item in list(collection):
                if item.users == 0:
                    collection.remove(item)

    def setup_lighting(self):
        bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
        sun = bpy.context.object
        sun.data.energy = 6.0
        sun.data.angle = 0.1
        
        bpy.ops.object.light_add(type='AREA', location=(-3, -3, 8))
        area1 = bpy.context.object
        area1.data.energy = 4.0
        area1.data.size = 4.0
        
        bpy.ops.object.light_add(type='AREA', location=(3, 3, 2))
        area2 = bpy.context.object
        area2.data.energy = 3.0
        area2.data.size = 3.0

    def setup_world(self):
        world = bpy.context.scene.world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()
        
        output = nodes.new('ShaderNodeOutputWorld')
        background = nodes.new('ShaderNodeBackground')
        background.inputs['Color'].default_value = (0.2, 0.2, 0.25, 1.0)
        background.inputs['Strength'].default_value = 0.8
        links.new(background.outputs[0], output.inputs[0])

    def load_material(self, kind: str, face: str):
        material_name = f"{kind}_{face}_mat"
        mat = bpy.data.materials.get(material_name)
        if not mat:
            mat = bpy.data.materials.new(material_name)
        
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        output = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        texture_path = self.textures_dir / kind / f"{face}.jpg"
        if texture_path.exists():
            tex_image = nodes.new('ShaderNodeTexImage')
            tex_image.image = bpy.data.images.load(str(texture_path))
            coord = nodes.new('ShaderNodeTexCoord')
            mapping = nodes.new('ShaderNodeMapping')
            
            links.new(coord.outputs['UV'], mapping.inputs['Vector'])
            links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])
            links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
        else:
            colors = {
                'group_box': (0.85, 0.75, 0.65, 1.0),
                'laptop': (0.9, 0.9, 0.95, 1.0),
                'tablet': (0.95, 0.95, 0.95, 1.0),
                'pallet': (0.8, 0.6, 0.4, 1.0)
            }
            bsdf.inputs['Base Color'].default_value = colors.get(kind, (0.8, 0.8, 0.8, 1.0))
        
        bsdf.inputs['Roughness'].default_value = 0.6
        return mat

    def create_pallet(self):
        bpy.ops.mesh.primitive_cube_add(size=1, location=self.pallet_pos)
        pallet = bpy.context.object
        pallet.name = "Pallet"
        pallet.scale = self.pallet_dims
        
        mesh = pallet.data
        while mesh.uv_layers:
            mesh.uv_layers.remove(mesh.uv_layers[0])
        uv_layer = mesh.uv_layers.new(name="UV")
        
        pallet.data.materials.clear()
        pallet.data.materials.append(self.load_material('pallet', 'top'))
        pallet.data.materials.append(self.load_material('pallet', 'long'))
        pallet.data.materials.append(self.load_material('pallet', 'short'))
        
        for poly in mesh.polygons:
            normal = poly.normal
            if abs(normal.z) > 0.9:
                poly.material_index = 0  # top
            elif abs(normal.y) > abs(normal.x):
                poly.material_index = 1  # long
            else:
                poly.material_index = 2  # short

    def create_box(self, kind: str, position: Vector, rotation: float = 0):
        dims = self.box_dims[kind]
        
        jitter_xy = Vector((
        random.uniform(-0.003, 0.003),
        random.uniform(-0.003, 0.003),
        0.0))
        
        spawn_pos = position + jitter_xy

        bpy.ops.mesh.primitive_cube_add(size=1, location=spawn_pos)
        box = bpy.context.object
        box.name = f"{kind}_{len([o for o in bpy.context.scene.objects if o.name.startswith(kind)])}"

        if rotation == 90:
            box.scale = Vector((dims.y/2, dims.x/2, dims.z/2))
            size_local = Vector((dims.y, dims.x, dims.z))
        else:
            box.scale = Vector((dims.x/2, dims.y/2, dims.z/2))
            size_local = Vector((dims.x, dims.y, dims.z))
        box.rotation_euler = Euler((0, 0, math.radians(rotation)), 'XYZ')

        me = box.data
        me.materials.clear()
        me.materials.append(self.load_material(kind, 'top'))   # 0
        me.materials.append(self.load_material(kind, 'long'))  # 1
        me.materials.append(self.load_material(kind, 'short')) # 2

        if not me.uv_layers:
            me.uv_layers.new(name="UVMap")
        uv = me.uv_layers.active

        def map_face(poly, ax_u, ax_v):
            for li in poly.loop_indices:
                vi = me.loops[li].vertex_index
                co = me.vertices[vi].co 
                u = co[ax_u] + 0.5
                v = co[ax_v] + 0.5
                uv.data[li].uv = (u, v)

        for poly in me.polygons:
            n = poly.normal
            ax = (abs(n.x), abs(n.y), abs(n.z))

            # Верх и низ — всегда top (плоскость XY)
            if ax[2] >= ax[0] and ax[2] >= ax[1]:
                poly.material_index = 0
                map_face(poly, 0, 1)
                continue

            horiz_long_is_x = size_local.x >= size_local.y

            if ax[1] > ax[0]:
                poly.material_index = 1 if horiz_long_is_x else 2  
                map_face(poly, 0, 2)  # U=X, V=Z
            else:
                poly.material_index = 1 if not horiz_long_is_x else 2  
                map_face(poly, 1, 2)  # U=Y, V=Z
                
        me.update()

    def optimize_placement(self, counts: Dict[str, int]) -> List[Tuple[str, Vector, float]]:
        """Оптимальное размещение коробок методом bin packing"""
        placements = []
        
        grid_size = 0.05
        x_positions = []
        y_positions = []
        
        x = -self.pallet_dims.x/2
        while x < self.pallet_dims.x/2:
            x_positions.append(x)
            x += grid_size
            
        y = -self.pallet_dims.y/2
        while y < self.pallet_dims.y/2:
            y_positions.append(y)
            y += grid_size
        
        height_map = {}
        for x in x_positions:
            for y in y_positions:
                height_map[(x, y)] = 0
        
        box_types = sorted(counts.keys(), 
                          key=lambda k: self.box_dims[k].x * self.box_dims[k].y * self.box_dims[k].z, 
                          reverse=True)
        
        for box_type in box_types:
            for _ in range(counts[box_type]):
                placed = False
               
                for rotation in [0, 90]:
                    if placed:
                        break
                        
                    dims = self.box_dims[box_type]
                    if rotation == 90:
                        w, d, h = dims.y, dims.x, dims.z
                    else:
                        w, d, h = dims.x, dims.y, dims.z
                    
                    best_pos = None
                    min_height = float('inf')
                    
                    for x in x_positions:
                        for y in y_positions:
                            if (x + w/2 <= self.pallet_dims.x/2 and 
                                y + d/2 <= self.pallet_dims.y/2 and
                                x - w/2 >= -self.pallet_dims.x/2 and
                                y - d/2 >= -self.pallet_dims.y/2):
                                
                                max_h = 0
                                for check_x in x_positions:
                                    if abs(check_x - x) <= w/2:
                                        for check_y in y_positions:
                                            if abs(check_y - y) <= d/2:
                                                max_h = max(max_h, height_map.get((check_x, check_y), 0))
                                
                                if max_h + h <= self.max_stack_height:
                                    if max_h < min_height:
                                        min_height = max_h
                                        best_pos = (x, y,max_h+ h/4, rotation)## 
                    
                    if best_pos:
                        x, y, z, rot = best_pos
                        position = Vector((x, y, z)) ### self.pallet_dims.z +
                        placements.append((box_type, position, rot))
                        
                        for check_x in x_positions:
                            if abs(check_x - x) <= w/2:
                                for check_y in y_positions:
                                    if abs(check_y - y) <= d/2:
                                        height_map[(check_x, check_y)] = min_height + h/2
                        
                        placed = True
                        break
                
                if not placed:
                    print(f"Не удалось разместить коробку {box_type}")
        
        return placements

    def setup_camera(self, side: str):
        pos = self.cameras[side]
        bpy.ops.object.camera_add(location=pos)
        camera = bpy.context.object
        
        direction = (Vector((0, 0, 0.2)) - pos).normalized()
        camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        
        camera.data.lens = 35
        camera.data.clip_start = 0.1
        camera.data.clip_end = 100
        
        bpy.context.scene.camera = camera
        return camera

    def render_scene(self, side: str, output_path: Path):
        camera = self.setup_camera(side)
        scene = bpy.context.scene
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        scene.render.filepath = str(output_path)
        
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(camera, do_unlink=True)

    def generate_single_scene(self, laptop_count: int, tablet_count: int, group_box_count: int, 
                             output_name: str = None):
        """Генерирует одну сцену с заданным количеством коробок"""
        counts = {
            'laptop': laptop_count,
            'tablet': tablet_count,
            'group_box': group_box_count
        }
        
        if output_name is None:
            output_name = f"pallet_{laptop_count}_{tablet_count}_{group_box_count}"
        
        self.clear_scene()
        self.setup_lighting()
        self.setup_world()
        self.create_pallet()
        
        placements = self.optimize_placement(counts)
        for box_type, position, rotation in placements:
            self.create_box(box_type, position, rotation)
        
        output_folder = self.output_dir / output_name
        output_folder.mkdir(exist_ok=True)
        
        for side in ['left', 'right']:
            self.render_scene(side, output_folder / f"{side}.png")
        
        metadata = {
            'counts': counts,
            'total_boxes': sum(counts.values()),
            'placements': len(placements),
            'box_dimensions': {k: list(v) for k, v in self.box_dims.items()}
        }
        
        with open(output_folder / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Сгенерирована сцена: {output_name}")
        return placements

    def generate_test_scenes(self):
        """Генерирует тестовые сцены для проверки"""
        test_configs = [
            (0, 8, 1),   # Много tablet + group_box
            (4, 12, 2),   # Полная загрузка
        ]
        
        self.setup_gpu_cycles()
        
        for i, (laptop, tablet, group_box) in enumerate(test_configs):
            output_name = f"pallet_{laptop}_{tablet}_{group_box}"
            self.generate_single_scene(laptop, tablet, group_box, output_name)

    def generate_dataset(self, dataset_size: int = 50):
        """Генерирует датасет заданного размера"""
        self.setup_gpu_cycles()
        
        for i in range(dataset_size):
            laptop_count = random.randint(0, 25)
            tablet_count = random.randint(0, 50)
            group_box_count = random.randint(0, 12)
            
            output_name = f"pallet_{laptop_count}_{tablet_count}_{group_box_count}"
            self.generate_single_scene(laptop_count, tablet_count, group_box_count, output_name)

if __name__ == "__main__":
    generator = PalletBoxGenerator()
#    print("Генерация тестовых сцен...")
#    generator.generate_test_scenes()
    
    print("Генерация датасета...")
    generator.generate_dataset(100)
