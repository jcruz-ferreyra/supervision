import math
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

from supervision.detection.core import Detections
from supervision.draw.color import Color
from supervision.geometry.core import Point, Rect, Vector

def _validate_direction(direction: str) -> None:
    is_valid = direction in ["in", "out", "both"]
    if not is_valid:
        raise ValueError("direction must be one of the following values: \"in\", \"out\", \"both\"")
    
def _validate_reference(reference: str) -> None:
    is_valid = reference in ["center", "top", "bottom", "full"]
    if not is_valid:
        raise ValueError("direction must be one of the following values: \"center\", \"top\", \"bottom\", \"full\"")

class LineZone:
    """
    Count the number of objects that cross a line.
    """

    def __init__(
            self,
            start: Point,
            end: Point,
            name: str = None,
            direction: str = "both",
            reference: str = "center"
            ):
        """
        Initialize a LineCounter object.

        Attributes:
            start (Point): The starting point of the line.
            end (Point): The ending point of the line.
            name (str): Name of the line_counter.
            direction (str): What count of items crossing the line is displayed.
                "in": Start point to the left, elements crossing up the line.
                "out": Start point to the left, elements crossing down the line.
                "both": Start point to the left, both "in" and "out" elements.
            reference (str): What part of the bbox is considered for crossing objects.
                "center": Center point of bbox.
                "top": Center point of bbox's top.
                "bottom": Center point of bbox's bottom.
                "full": Four bbox's anchors.
        
        """
        self.name = name
        self.vector = Vector(start=start, end=end)
        self.tracker_state: Dict[str, bool] = {}
        self.in_count: Dict[str, int] = {}
        self.out_count: Dict[str, int] = {}
        self.counted_trackers: List[int] = []
        self.direction: str = direction
        self.reference: bool = reference

        _validate_direction(direction=self.direction)
        _validate_reference(reference=self.reference)

    def trigger(self, detections: Detections):
        """
        Update the in_count and out_count for the detections that cross the line.

        Attributes:
            detections (Detections): The detections for which to update the counts.

        """
        for xyxy, _, confidence, class_id, tracker_id in detections:
            # handle detections with no tracker_id
            if tracker_id is None:
                continue
            elif tracker_id in self.counted_trackers:
                continue

            x1, y1, x2, y2 = xyxy

            if self.reference == "center":
                x = (x1 + x2) // 2
                y = (y1 + y2) // 2
                anchor = Point(x=x, y=y)
                tracker_state = self.vector.is_in(point=anchor)
            elif self.reference == "top":
                x = (x1 + x2) // 2
                y = y1
                anchor = Point(x=x, y=y)
                tracker_state = self.vector.is_in(point=anchor)
            elif self.reference == "bottom":
                x = (x1 + x2) // 2
                y = y2
                anchor = Point(x=x, y=y)
                tracker_state = self.vector.is_in(point=anchor)
            elif self.reference == "full":
                # we check if all four anchors of bbox are on the same side of vector
                anchors = [
                    Point(x=x1, y=y1),
                    Point(x=x1, y=y2),
                    Point(x=x2, y=y1),
                    Point(x=x2, y=y2),
                ]
                triggers = [self.vector.is_in(point=anchor) for anchor in anchors]

                # detection is partially in and partially out
                if len(set(triggers)) == 2:
                    continue

                # we get if the whole bbox is in or out with respect to the line counter
                tracker_state = triggers[0]

            # handle new detection
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                continue

            # handle detection on the same side of the line
            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            # handle detection that crossed the line
            self.tracker_state[tracker_id] = tracker_state
            if tracker_state:
                if class_id in self.in_count:
                    self.in_count[class_id] += 1
                else:
                    self.in_count[class_id] = 1
                self.counted_trackers.append(tracker_id)
            else:
                if class_id in self.out_count:
                    self.out_count[class_id] += 1
                else:
                    self.out_count[class_id] = 1
                self.counted_trackers.append(tracker_id)


class LineZoneAnnotator:
    def __init__(
        self,
        thickness: float = 2,
        color: Color = Color.white(),
        text_thickness: float = 2,
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_offset: float = 1.5,
        text_padding: int = 10,
        custom_in_text: Optional[str] = None,
        custom_out_text: Optional[str] = None,
    ):
        """
        Initialize the LineCounterAnnotator object with default values.

        Attributes:
            thickness (float): The thickness of the line that will be drawn.
            color (Color): The color of the line that will be drawn.
            text_thickness (float): The thickness of the text that will be drawn.
            text_color (Color): The color of the text that will be drawn.
            text_scale (float): The scale of the text that will be drawn.
            text_offset (float): The offset of the text that will be drawn.
            text_padding (int): The padding of the text that will be drawn.

        """
        self.thickness: float = thickness
        self.color: Color = color
        self.text_thickness: float = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding
        self.custom_in_text: str = custom_in_text
        self.custom_out_text: str = custom_out_text

    def annotate(self, frame: np.ndarray, line_counter: LineZone) -> np.ndarray:
        """
        Draws the line on the frame using the line_counter provided.

        Attributes:
            frame (np.ndarray): The image on which the line will be drawn.
            line_counter (LineCounter): The line counter that will be used to draw the line.

        Returns:
            np.ndarray: The image with the line drawn on it.

        """

        def annotate_count(text, text_over=True):
            """
            Draws the counter for in/out counts aligned to the line.

            Attributes:
                text (str): Line of text to annotate alongside the count.
                text_over (Bool): Position of the text over/under the line.

            Returns:
                np.ndarray: Frame with the count annotated in it.
            """
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
            )
            background_dim = max(text_width, text_height) + 30

            # Create squared background images to place text and text box.
            box_background = np.zeros((background_dim, background_dim), dtype=np.uint8)
            text_background = np.zeros((background_dim, background_dim), dtype=np.uint8)

            text_position = (
                (background_dim // 2) - (text_width // 2),
                (background_dim // 2) + (text_height // 2),
            )

            # Draw text box.
            text_box_background = Rect(
                x=text_position[0],
                y=text_position[1] - text_height,
                width=text_width,
                height=text_height,
            ).pad(padding=self.text_padding)

            cv2.rectangle(
                box_background,
                text_box_background.top_left.as_xy_int_tuple(),
                text_box_background.bottom_right.as_xy_int_tuple(),
                (255, 255, 255),
                -1,
            )

            # Draw text.
            cv2.putText(
                text_background,
                text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_scale,
                (255, 255, 255),
                self.text_thickness,
                cv2.LINE_AA,
            )

            # Rotate text and text box.
            start_point = line_counter.vector.start.as_xy_int_tuple()
            end_point = line_counter.vector.end.as_xy_int_tuple()

            try:
                line_angle = math.degrees(
                    math.atan(
                        (end_point[1] - start_point[1])
                        / (end_point[0] - start_point[0])
                    )
                )
                if (end_point[0] - start_point[0]) < 0:
                    line_angle = 180 + line_angle
            except ZeroDivisionError:
                line_angle = 90
                if (end_point[1] - start_point[1]) < 0:
                    line_angle = 180 + line_angle

            rotation_center = ((background_dim // 2), (background_dim // 2))
            rotation_angle = -(line_angle)
            rotation_scale = 1
            rotation_matrix = cv2.getRotationMatrix2D(
                rotation_center, rotation_angle, rotation_scale
            )

            box_background_rotated = cv2.warpAffine(
                box_background, rotation_matrix, (background_dim, background_dim)
            )
            text_background_rotated = cv2.warpAffine(
                text_background, rotation_matrix, (background_dim, background_dim)
            )

            # Set position of the text along and perpendicular to the line.
            text_insertion = list(end_point)

            move_along_x = int(
                math.cos(math.radians(line_angle))
                * (text_width / 2 + self.text_padding)
            )
            move_along_y = int(
                math.sin(math.radians(line_angle))
                * (text_width / 2 + self.text_padding)
            )

            move_perp_x = int(
                math.sin(math.radians(line_angle))
                * (text_height / 2 + self.text_padding * 2)
            )
            move_perp_y = int(
                math.cos(math.radians(line_angle))
                * (text_height / 2 + self.text_padding * 2)
            )

            text_insertion[0] -= move_along_x
            text_insertion[1] -= move_along_y
            if text_over:
                text_insertion[0] += move_perp_x
                text_insertion[1] -= move_perp_y
            else:
                text_insertion[0] -= move_perp_x
                text_insertion[1] += move_perp_y

            # Trim pixels of text and pixels of text box that are out of the frame.
            y1 = max(text_insertion[1] - background_dim // 2, 0)
            y2 = min(
                text_insertion[1] + background_dim // 2 + background_dim % 2,
                frame.shape[0],
            )
            x1 = max(text_insertion[0] - background_dim // 2, 0)
            x2 = min(
                text_insertion[0] + background_dim // 2 + background_dim % 2,
                frame.shape[1],
            )

            if y2 - y1 != background_dim:
                if y1 == 0:
                    box_background_rotated = box_background_rotated[
                        (background_dim - y2) :, :
                    ]
                    text_background_rotated = text_background_rotated[
                        (background_dim - y2) :, :
                    ]
                elif y2 == frame.shape[0]:
                    box_background_rotated = box_background_rotated[: (y2 - y1), :]
                    text_background_rotated = text_background_rotated[: (y2 - y1), :]

            if x2 - x1 != background_dim:
                if x1 == 0:
                    box_background_rotated = box_background_rotated[
                        :, (background_dim - x2) :
                    ]
                    text_background_rotated = text_background_rotated[
                        :, (background_dim - x2) :
                    ]
                elif x2 == frame.shape[1]:
                    box_background_rotated = box_background_rotated[:, : (x2 - x1)]
                    text_background_rotated = text_background_rotated[:, : (x2 - x1)]

            # Annotate text and text box to orignal frame.
            frame[y1:y2, x1:x2, 0][box_background_rotated > 95] = self.color.as_bgr()[0]
            frame[y1:y2, x1:x2, 1][box_background_rotated > 95] = self.color.as_bgr()[1]
            frame[y1:y2, x1:x2, 2][box_background_rotated > 95] = self.color.as_bgr()[2]

            frame[y1:y2, x1:x2, 0][
                text_background_rotated != 0
            ] = self.text_color.as_bgr()[0] * (
                text_background_rotated[text_background_rotated != 0] / 255
            ) + self.color.as_bgr()[
                0
            ] * (
                1 - (text_background_rotated[text_background_rotated != 0] / 255)
            )
            frame[y1:y2, x1:x2, 1][
                text_background_rotated != 0
            ] = self.text_color.as_bgr()[1] * (
                text_background_rotated[text_background_rotated != 0] / 255
            ) + self.color.as_bgr()[
                1
            ] * (
                1 - (text_background_rotated[text_background_rotated != 0] / 255)
            )
            frame[y1:y2, x1:x2, 2][
                text_background_rotated != 0
            ] = self.text_color.as_bgr()[2] * (
                text_background_rotated[text_background_rotated != 0] / 255
            ) + self.color.as_bgr()[
                2
            ] * (
                1 - (text_background_rotated[text_background_rotated != 0] / 255)
            )

            return frame

        # Draw line.
        cv2.line(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            line_counter.vector.end.as_xy_int_tuple(),
            self.color.as_bgr(),
            self.thickness,
            lineType=cv2.LINE_AA,
            shift=0,
        )
        cv2.circle(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            radius=self.thickness,
            color=self.color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        # Create in/out text.
        in_text = (
            f"{self.custom_in_text}: {line_counter.in_count}"
            if self.custom_in_text is not None
            else f"in: {line_counter.in_count}"
        )
        out_text = (
            f"{self.custom_out_text}: {line_counter.out_count}"
            if self.custom_out_text is not None
            else f"out: {line_counter.out_count}"
        )

        if line_counter.direction == "both":
            frame = annotate_count(in_text, text_over=True)
            frame = annotate_count(out_text, text_over=False)
        elif line_counter.direction == "in":
            frame = annotate_count(in_text, text_over=True)
        elif line_counter.direction == "out":
            frame = annotate_count(out_text, text_over=False)
        else:
            pass

        return frame
