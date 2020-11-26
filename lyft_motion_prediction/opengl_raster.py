import numpy as np

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.data.filter import (filter_agents_by_labels, filter_agents_by_track_id)
from l5kit.data.map_api import MapAPI
from l5kit.dataset import AgentDataset
from l5kit.rasterization import Rasterizer
from l5kit.rasterization.box_rasterizer import get_ego_as_agent
from l5kit.rasterization.rasterizer_builder import (_load_metadata, get_hardcoded_world_to_ecef)

from matplotlib import pyplot as plt
from opengl import *
from opengl.entities import Agent
from OpenGL.GL import *
from OpenGL.GLUT import *
from tqdm import tqdm
from typing import List, Optional, Tuple, Union


class OpenGLSemanticRasterizer(Rasterizer):

    def __init__(self, raster_size: Tuple[int, int], pixel_size: Union[np.ndarray, list, float], ego_center: np.ndarray,
                 filter_agents_threshold: float, history_num_frames: int, semantic_map_path: str,
                 world_to_ecef: np.ndarray):
        super().__init__()

        self.raster_size = raster_size
        self.pixel_size = pixel_size

        if isinstance(pixel_size, np.ndarray) or isinstance(pixel_size, list):
            self.pixel_size = pixel_size[0]

        self.filter_agents_threshold = filter_agents_threshold

        self.proto_API = MapAPI(semantic_map_path, world_to_ecef)

        # Create and hide a window
        DisplayManager.create_display(width=raster_size[0], height=raster_size[1])

        self.output_fbo = initialize_framebuffer_object(width=raster_size[0], height=raster_size[1])

        self.loader: Loader = Loader()
        self.shader: StaticShader = StaticShader()
        self.camera: Camera = Camera()
        self.renderer: Renderer = Renderer(
            display_width=raster_size[0],
            display_height=raster_size[1],
            camera=self.camera,
            pixel_size=self.pixel_size
        )

        self.renderer.add_renderer('entity_renderer', EntityRenderer())
        self.renderer.add_renderer('map_renderer', MapRenderer())

        self.agent_model: Cube = Cube()
        self.agent_model.load_to_vao(self.loader)

        self.lane_surface_model: LaneSurfaceModel = create_lane_surface_model(self.proto_API)
        self.lane_surface_model.load_to_vao(self.loader)

        self.lane_lines_model: LaneLinesModel = create_lane_line_model(self.proto_API)
        self.lane_lines_model.load_to_vao(self.loader)

        self.crosswalk_model: CrosswalkModel = create_crosswalks_model(self.proto_API)
        self.crosswalk_model.load_to_vao(self.loader)

        self.agents = []
        self.map_layers = []

    def add_agents(self, agents: Tuple[np.ndarray, bool]):

        ego = agents[1]

        for agent_data in agents[0]:

            agent: Agent = Agent(
                model=self.agent_model,
                position=np.array([agent_data["centroid"][0], agent_data["centroid"][1], 0.]),
                rotation=np.array([0., 0., agent_data["yaw"]]),
                scale=np.array([agent_data["extent"][0], agent_data["extent"][1], 0.]),
                color=np.array([0., 1., 0.] if ego else [0., 0., 1.], dtype=np.float32)
            )

            self.agents.append(agent)

    def rasterize(self, history_frames: np.ndarray, history_agents: List[np.ndarray],
                  history_tl_faces: List[np.ndarray], agent: Optional[np.ndarray] = None,
                  agents: Optional[np.ndarray] = None
                  ) -> np.ndarray:

        frame = history_frames[0]
        if agent is None:
            translation = frame["ego_translation"]
            yaw = frame["ego_rotation"]
        else:
            translation = agent["centroid"]
            yaw = agent["yaw"]

        self.agents = []
        self.camera.position = np.array([translation[0], translation[1], 1.], dtype=np.float32)
        self.camera.rotation = np.array([0., 0., yaw], dtype=np.float32)

        # Actor/Ego rasterizer
        for i, (frame, agents) in enumerate(zip(history_frames, history_agents)):
            agents = filter_agents_by_labels(agents, self.filter_agents_threshold)
            # note the cast is for legacy support of dataset before April 2020
            av_agent = get_ego_as_agent(frame).astype(agents.dtype)

            if agent is None:
                self.add_agents((agents, False))
                self.add_agents((av_agent, True))
            else:
                agent_ego = filter_agents_by_track_id(agents, agent["track_id"])

                if len(agent_ego) == 0:  # agent not in this history frame
                    self.add_agents((agents, False))
                    self.add_agents((av_agent, False))
                else:
                    agents = agents[agents != agent_ego[0]]
                    self.add_agents((agents, False))
                    self.add_agents((av_agent, False))
                    self.add_agents((agent_ego, True))

            # TODO: history frames.
            break

        glBindFramebuffer(GL_FRAMEBUFFER, self.output_fbo)

        self.renderer.render([
            ('map_renderer', [
                # Map layer #1 - road surface
                self.lane_surface_model,
                # Map layer #2 - crosswalks,
                self.crosswalk_model,
                # Map layer #3 - lane lines
                self.lane_lines_model,
            ]),
            ('entity_renderer', self.agents),
        ])

        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        image = glReadPixels(0, 0, self.raster_size[0], self.raster_size[1], GL_RGB, GL_FLOAT)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        return image

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return in_im


if __name__ == '__main__':
    os.environ["L5KIT_DATA_FOLDER"] = "./input"
    config_file = "baseline_agent_motion.yaml"
    cfg = load_config_data(f"./configs/{config_file}")

    dm = LocalDataManager()
    dataset_path = dm.require(cfg["train_data_loader"]["key"])
    zarr_dataset = ChunkedDataset(dataset_path)
    zarr_dataset.open()

    raster_cfg = cfg["raster_params"]
    semantic_map_filepath = dm.require(raster_cfg["semantic_map_key"])
    try:
        dataset_meta = _load_metadata(raster_cfg["dataset_meta_key"], dm)
        world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
    except (KeyError, FileNotFoundError):
        world_to_ecef = get_hardcoded_world_to_ecef()

    rast = OpenGLSemanticRasterizer(
        raster_size=raster_cfg["raster_size"],
        pixel_size=raster_cfg["pixel_size"],
        ego_center=raster_cfg["ego_center"],
        filter_agents_threshold=0.5,
        history_num_frames=0,
        semantic_map_path=semantic_map_filepath,
        world_to_ecef=world_to_ecef,
    )
    dataset = AgentDataset(cfg, zarr_dataset, rast)

    for i in tqdm(range(3200)):
        data = dataset[np.random.randint(0, len(dataset))]

    # Show an example
    data = dataset[7500]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)

    # cv2.imwrite('test_opengl.png', im[::-1])
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(im[::-1])
    plt.show()