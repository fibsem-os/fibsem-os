from fibsem.structures import FibsemStagePosition, FibsemRectangle
from fibsem.modules_czii.Pipeline_CoincidenceSetup import CoincidenceSetup
from fibsem.fm.structures import ChannelSettings, ZParameters, FocusMethod, FluorescenceImage, AutoFocusSettings, ZStackOrder
from fibsem.fm.calibration import run_autofocus, run_coarse_fine_autofocus
from fibsem.fm.structures import ChannelSettings, ZParameters
from fibsem.fm.acquisition import acquire_channels, acquire_z_stack, acquire_image
import numpy as np
import tkinter as tk
from ultralytics import YOLO
import os
from tkinter import messagebox

class CLEMFunctions:
    def __init__(self, sample_type, oa, milling_angle, channel_settings_list=None):
        self.oa = oa
        self.sample_type = sample_type
        self.milling_angle = milling_angle
        self.channel_settings_list = channel_settings_list
        from fibsem.modules_czii.Image_Acquisition import ImageAcquisition
        self.imaging = ImageAcquisition(oa=self.oa)
        self.coincidence_setup = CoincidenceSetup(oa=oa, sample_type=self.sample_type, milling_angle=self.milling_angle)

    def mill_clem_label(self):
        self.oa.fl_microscope.objective.retract()
        
        self.oa.pop_up_message(title='Move to position', message='Please move to a suitable location for the CLEM label and confirm with OK.')
        #self.coincidence_setup.run_coincidence_setup()
        grid_start_position = self.oa.fib_microscope.get_stage_position()
        
        grid_backside = FibsemStagePosition(x=grid_start_position.x, y=grid_start_position.y, z=grid_start_position.z, t=np.deg2rad(-128.0), coordinate_system='RAW')
        print("[INFO] Move the backside of the grid perpendicular to the FIB beam.")
        self.oa.fib_microscope.safe_absolute_stage_movement(grid_backside)
        rules = [{"task": "tilt", "op": "==", "threshold": -128},]
        if not self.oa.wait_for_correct_microscope_status(rules, message="Please set tilt to -128°.",):
            return
        
        from fibsem.modules_czii.Milling import CustomMillingPattern
        self.milling = CustomMillingPattern(oa=self.oa, sample_type=self.sample_type)
        print("[INFO] Milling CLEM reference label.")
        self.milling.run_milling(stage='clem_label')
        self.oa.fib_microscope.safe_absolute_stage_movement(grid_start_position)
        

        
    def adjust_z_stage_offset(self, lamella_z_position=0.0e-6):
        current_stage_position = self.oa.fib_microscope.get_stage_position()
        delta_z = current_stage_position.z - lamella_z_position
        if self.milling_angle == -15:
            shift_y = -0.22147 * delta_z
        elif self.milling_angle == -18:
            shift_y = -0.264 * delta_z
        else:
            print("[WARNING] Milling angle is not setup.")
            shift_y = 0
        self.oa.fib_microscope.move_stage_relative(FibsemStagePosition(y=shift_y))
        
    def acquire_clem_data(self, path=None, name_token=None, lamella_z_position=0.0e-6):
        self.adjust_z_stage_offset(lamella_z_position=lamella_z_position)

        best_focus_fl = self.focus_on_lamella(path=path, name_token=name_token)

        channel_settings = [ChannelSettings(
                                            name="Channel-01",
                                            excitation_wavelength=650,
                                            emission_wavelength=None,
                                            exposure_time=0.005,
                                            power=0.01,
                                            gain=0.0,)]
        
        for i, channel in enumerate(self.channel_settings_list[1:]):
            settings = channel.to_dict()
            exitation_wavelength = settings['excitation_wavelength']
            emission_wavelength = settings['emission_wavelength']
            exposure_time = settings['exposure_time']
            power = settings['power']
            gain = settings['gain'] 
            
            channel_settings.append(
                    ChannelSettings(
                                    name=f"Channel-0{i+1}",
                                    excitation_wavelength=exitation_wavelength,
                                    emission_wavelength=emission_wavelength,
                                    exposure_time=exposure_time,
                                    power=power,
                                    gain=gain,))
                                
        zstack_settings = {'z_min': -2.0e-6, 'z_max': 2.0e-6, 'z_step': 0.5e-6}
        if best_focus_fl is not None:
            self.imaging.acquire_zstack(channel_settings=channel_settings, zstack_settings=zstack_settings, path=path, tilt=-180+self.milling_angle, binning=4, order='objective_pos', name_token=f"{name_token}_clem_data", focus_preset=best_focus_fl)
        else:
            print(['ERROR NOT YET IMPLEMENTED.'])


    
    def acquire_clem_data_backup(self, path=None, best_focus=None, name_token=None):
        channel_settings_list_edited = [ChannelSettings(
                                            name="Channel-01",
                                            excitation_wavelength=550,
                                            emission_wavelength=None,
                                            exposure_time=0.075,
                                            power=0.01,
                                             gain=0.0,)]

        self.oa.fl_microscope.set_binning(2)

        for i, channel in enumerate(self.channel_settings_list[1:]):
            settings = channel.to_dict()
            exitation_wavelength = settings['excitation_wavelength']
            emission_wavelength = settings['emission_wavelength']
            exposure_time = settings['exposure_time']*5
            power = settings['power']
            if settings['gain'] > 0 and settings['gain'] < 0.5:
                gain = 0.5
            else:
                gain = settings['gain'] * 1.5

            channel_settings_list_edited.append(
                    ChannelSettings(
                        name=f"Channel-0{i+1}",
                        excitation_wavelength=exitation_wavelength,
                        emission_wavelength=emission_wavelength,
                        exposure_time=exposure_time,
                        power=power,
                        gain=gain,))
            
        zstack_settings = {'z_min': -3.0e-6, 'z_max': 3.0e-6, 'z_step': 0.5e-6}
        if best_focus is not None:
            self.imaging.acquire_zstack(channel_settings=channel_settings_list_edited, zstack_settings=zstack_settings, path=path, tilt=-180+self.milling_angle, binning=1, order='objective_pos', name_token=f"{name_token}_clem_data", focus_preset=best_focus)
        else:
            print(['ERROR NOT YET IMPLEMENTED.'])
        
        
