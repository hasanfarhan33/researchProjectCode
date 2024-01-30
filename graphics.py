import glob 
import os 
import tempfile 
import carla 
import imageio 
import matplotlib.pyplot as plt 
import numpy as np 
import pygame 
import tqdm 
from absl import logging 
from skimage import transform 

IM_WIDTH = 400 
IM_HEIGHT = 400

def setup(width: int = IM_WIDTH, height: int = IM_HEIGHT, render: bool = True):
    # Pygame setup 
    pygame.init()
    pygame.display.set_caption("OATomobile")
    
    if render: 
        logging.debug("Pygame initializes a window display")
        display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF, )
    else: 
        logging.debug("Pygame initializes a headless display")
        display = pygame.Surface((width, height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("dejavusansmono", 14)
    return display, clock, font 

def make_dashboard(display, font, clock, observations) -> None: 
    """Generates a dashboard used for visualizing the agent"""
    
    # Clearn dashboard 
    display.fill(COLORS["BLACK"])
    
    # Adaptive Width 
    ada_width = 0 
    
    if "preview_camera" in observations:
        # Render the chase camera view 
        ob_chase_cam_rgb = ndarray_to_pygame_surface(
            array = observations.get("preview_camera"), 
            swapaxes = True 
        )   
        display.blit(ob_chase_cam_rgb, (ada_width, 0))
        ada_width = ada_width + ob_chase_cam_rgb.get_width() 
        
def ndarray_to_pygame_surface(array, swapaxes): 
    """Returns a pygame surface from a numpy array"""
    # Make sure it is in 255 range 
    array = 255 * (array / array.max())
    if swapaxes: 
        array = array.swapaxes(0, 1)
    return pygame.surfarray.make_surface(array)

# Color palette
COLORS = {
    "WHITE": pygame.Color(255, 255, 255),
    "BLACK": pygame.Color(0, 0, 0),
    "RED": pygame.Color(255, 0, 0),
    "GREEN": pygame.Color(0, 255, 0),
    "BLUE": pygame.Color(0, 0, 255),
    "SILVER": pygame.Color(195, 195, 195),
}