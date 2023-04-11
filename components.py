"""Module for loading data from disk."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import asyncio
from limbus.core import Component, InputParams, OutputParams, ComponentState

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import cv2
import numpy as np
import torch
import kornia as K
from kornia.core import Tensor, concatenate
import kornia_rs as KR

from farm_ng.core import event_pb2
from farm_ng.core.events_file_reader import EventLogPosition
from farm_ng.core.events_file_reader import EventsFileReader
from farm_ng.oak import oak_pb2


class EventsFileReaderComponent(Component):
    def __init__(self, name: str, events_file: str | Path, device: str) -> None:
        super().__init__(name)
        reader = EventsFileReader(events_file)
        succeeded: bool = reader.open()
        if not succeeded:
            raise Exception(f"Failed to open events file: {events_file}")
        
        # filter the events containing `oak_pb2.OakDataSample`
        self.events: list[EventLogPosition] = [
            x for x in reader.get_index() if self._event_has_message(x.event, oak_pb2.OakDataSample)
        ]
        self.event_idx: int = 0

        # read data
        self.image_decoder = KR.ImageDecoder()
        self.device = device

        # register outputs
        cams_uris = set([x.event.uri.path.split("/")[0] for x in self.events])
        for cam_uri in cams_uris:
            self.outputs.declare(cam_uri)
    
    def _event_has_message(self, event: event_pb2.Event, msg_type) -> bool:
        """Returns true if the event contains a message of the given type.
        
        Args:
            event: The event to check.
            msg_type: The type of the message to check for.
        """
        return event.uri.query.split("&")[0].split(".")[-1] == msg_type.__name__
    
    def _decode_jpeg(self, data: bytes) -> Tensor:
        """Decodes a JPEG image.
        
        Args:
            data: The JPEG image data.
        """
        dlpack = self.image_decoder.decode(data)
        img_np = np.from_dlpack(dlpack)
        img_t = K.utils.image_to_tensor(img_np.copy(), keepdim=True)
        img_t.to(self.device)
        img_t = K.geometry.resize(img_t.float(), size=1024, side="horz").byte()
        return img_t
    
    async def forward(self) -> ComponentState:
        if self.counter >= len(self.events):
            return ComponentState.STOPPED
        # get the next event
        event_log: EventLogPosition = self.events[self.event_idx]
        self.event_idx += 1

        # read the event data
        sample = event_log.read_message()
        frame: oak_pb2.OakSyncFrame = sample.frame

        # decode the image data and send it to the next component
        camera_name: str = event_log.event.uri.path.split("/")[0]
        await getattr(self.outputs, camera_name).send(
            self._decode_jpeg(frame.rgb.image_data)
        )
        return ComponentState.OK


class OpencvWindow(Component):

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        inputs.declare("image")
        inputs.declare("mask")

    async def forward(self):
        img: Tensor
        mask: Tensor
        img, mask = await asyncio.gather(
            self.inputs.image.receive(),
            self.inputs.mask.receive())

        img_vis = concatenate([img, mask], dim=-1)
        img_vis = K.color.rgb_to_bgr(img_vis)

        cv2.namedWindow(f"{self.name}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"{self.name}", K.utils.tensor_to_image(img_vis))
        cv2.waitKey(1)
        return ComponentState.OK


@dataclass
class ImageSegmentatorConfig:
    model_path: Path 
    model_uri: str = "sam/vit_h"
    model_checkpoint: str = "sam_vit_h_4b8939.pth"
    device: str = "cpu"


class ImageSegmentator:
    def __init__(self, config: ImageSegmentatorConfig) -> None:
        if "sam" in config.model_uri:
            # load the segmentation model
            model_type: str = config.model_uri.split("/")[-1]
            sam_checkpoint: Path = config.model_path / config.model_checkpoint
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(config.device)
            sam = torch.compile(sam)
            self.algorithm = SamAutomaticMaskGenerator(
                model=sam,
                #crop_n_layers=0,
                #min_mask_region_area=10
            )
        else:
            raise Exception(f"unsupported model: {config.model_uri}")
        
        self.color_map = torch.randint(
            0, 255, size=(256, 3), dtype=torch.uint8)

    def segment_image(self, image: Tensor) -> Tensor:
        t0 = time.time() * 1000
        image_np = K.utils.tensor_to_image(image)
        results = self.algorithm.generate(image_np)
        t1 = time.time() * 1000
        print(f"segmentation time: {t1 - t0}")
        mask = torch.zeros_like(image, dtype=torch.uint8, device=image.device)
        for class_id, result in enumerate(results):
            segmentation = result["segmentation"]
            seg_t = torch.tensor(segmentation).repeat(3, 1, 1).to(image.device)
            mask += self.color_map[class_id].view(3, 1, 1) * seg_t.byte()
        t2 = time.time() * 1000
        print(f"mask time: {t2 - t1}")
        print(f"total time: {t2 - t0}")
        print("--------------------------------------------------")
        return mask


class ImageSegmentatorComponent(Component):
    def __init__(self, name: str, model: ImageSegmentator) -> None:
        super().__init__(name)
        self.model = model

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        inputs.declare("image")
    
    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        outputs.declare("mask")
    
    async def forward(self) -> ComponentState:
        img: Tensor = await self.inputs.image.receive()
        mask: Tensor = self.model.segment_image(img)
        await self.outputs.mask.send(mask)
        return ComponentState.OK
