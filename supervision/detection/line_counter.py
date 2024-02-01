import math
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from supervision.detection.core import Detections
from supervision.draw.color import Color
from supervision.draw.utils import draw_text
from supervision.geometry.core import Point, Position, Rect, Vector


class LineZone:
    """
    Count the number of objects that cross a line.

        This class is responsible for counting the number of objects that cross a
    predefined line.

    !!! warning

        LineZone uses the `tracker_id`. Read
        [here](https://supervision.roboflow.com/trackers/) to learn how to plug
        tracking into your inference pipeline.

    Attributes:
        in_count (int): The number of objects that have crossed the line from outside
            to inside.
        out_count (int): The number of objects that have crossed the line from inside
            to outside.
    """

    def __init__(
        self,
        start: Point,
        end: Point,
        triggering_anchors: Iterable[Position] = (
            Position.TOP_LEFT,
            Position.TOP_RIGHT,
            Position.BOTTOM_LEFT,
            Position.BOTTOM_RIGHT,
        ),
    ):
        """
        Args:
            start (Point): The starting point of the line.
            end (Point): The ending point of the line.
            trigger_in (bool): Count object crossing in the line.
            trigger_out (bool): Count object crossing out the line.
            triggering_anchors (List[sv.Position]): A list of positions
                specifying which anchors of the detections bounding box
                to consider when deciding on whether the detection
                has passed the line counter or not. By default, this
                contains the four corners of the detection's bounding box
        """
        self.vector = Vector(start=start, end=end)
        self.limits = self.calculate_region_of_interest_limits(vector=self.vector)
        self.tracker_state: Dict[str, bool] = {}
        self.in_count: int = 0
        self.out_count: int = 0
        self.triggering_anchors = triggering_anchors

    @staticmethod
    def calculate_region_of_interest_limits(vector: Vector) -> Tuple[Vector, Vector]:
        magnitude = vector.magnitude

        if magnitude == 0:
            raise ValueError("The magnitude of the vector cannot be zero.")

        delta_x = vector.end.x - vector.start.x
        delta_y = vector.end.y - vector.start.y

        unit_vector_x = delta_x / magnitude
        unit_vector_y = delta_y / magnitude

        perpendicular_vector_x = -unit_vector_y
        perpendicular_vector_y = unit_vector_x

        start_region_limit = Vector(
            start=vector.start,
            end=Point(
                x=vector.start.x + perpendicular_vector_x,
                y=vector.start.y + perpendicular_vector_y,
            ),
        )
        end_region_limit = Vector(
            start=vector.end,
            end=Point(
                x=vector.end.x - perpendicular_vector_x,
                y=vector.end.y - perpendicular_vector_y,
            ),
        )
        return start_region_limit, end_region_limit

    @staticmethod
    def is_point_in_limits(point: Point, limits: Tuple[Vector, Vector]) -> bool:
        cross_product_1 = limits[0].cross_product(point)
        cross_product_2 = limits[1].cross_product(point)
        return (cross_product_1 > 0) == (cross_product_2 > 0)

    def trigger(self, detections: Detections) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the `in_count` and `out_count` based on the objects that cross the line.

        Args:
            detections (Detections): A list of detections for which to update the
                counts.

        Returns:
            A tuple of two boolean NumPy arrays. The first array indicates which
                detections have crossed the line from outside to inside. The second
                array indicates which detections have crossed the line from inside to
                outside.
        """
        crossed_in = np.full(len(detections), False)
        crossed_out = np.full(len(detections), False)

        if len(detections) == 0:
            return crossed_in, crossed_out

        all_anchors = np.array(
            [
                detections.get_anchors_coordinates(anchor)
                for anchor in self.triggering_anchors
            ]
        )

        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue

            box_anchors = [Point(x=x, y=y) for x, y in all_anchors[:, i, :]]

            in_limits = all(
                [
                    self.is_point_in_limits(point=anchor, limits=self.limits)
                    for anchor in box_anchors
                ]
            )

            if not in_limits:
                continue

            triggers = [
                self.vector.cross_product(point=anchor) > 0 for anchor in box_anchors
            ]

            if len(set(triggers)) == 2:
                continue

            tracker_state = triggers[0]

            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                continue

            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            self.tracker_state[tracker_id] = tracker_state
            if tracker_state:
                self.in_count += 1
                crossed_in[i] = True
            else:
                self.out_count += 1
                crossed_out[i] = True

        return crossed_in, crossed_out


class LineZoneAnnotator:
    def __init__(
        self,
        thickness: float = 2,
        color: Color = Color.WHITE,
        text_thickness: float = 2,
        text_color: Color = Color.BLACK,
        text_scale: float = 0.5,
        text_offset: float = 1.5,
        text_padding: int = 10,
        custom_in_text: Optional[str] = None,
        custom_out_text: Optional[str] = None,
        display_in_count: bool = True,
        display_out_count: bool = True,
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
            display_in_count (bool): Whether to display the in count or not.
            display_out_count (bool): Whether to display the out count or not.

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
        self.display_in_count: bool = display_in_count
        self.display_out_count: bool = display_out_count

    def annotate(self, frame: np.ndarray, line_counter: LineZone) -> np.ndarray:
        """
        Draws the line on the frame using the line_counter provided.

        Attributes:
            frame (np.ndarray): The image on which the line will be drawn.
            line_counter (LineCounter): The line counter
                that will be used to draw the line.

        Returns:
            np.ndarray: The image with the line drawn on it.

        """
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

        if self.display_in_count:
            in_text = (
                f"{self.custom_in_text}: {line_counter.in_count}"
                if self.custom_in_text is not None
                else f"in: {line_counter.in_count}"
            )
            frame = self._annotate_count(
                frame=frame, line_counter=line_counter, text=in_text, is_in_count=True
            )

        if self.display_out_count:
            out_text = (
                f"{self.custom_out_text}: {line_counter.out_count}"
                if self.custom_out_text is not None
                else f"out: {line_counter.out_count}"
            )
            frame = self._annotate_count(
                frame=frame, line_counter=line_counter, text=out_text, is_in_count=False
            )

        return frame

    def _annotate_count(
        self,
        frame: np.ndarray,
        line_counter: LineZone,
        text: str,
        is_in_count: bool,
    ) -> None:
        """
        Draws the counter for in/out counts aligned to the line.

        Args:
            frame (np.ndarray): The image on which the text will be drawn.
            line_counter (LineCounter): The line counter
                that will be used to draw the line.
            text (str): The text that will be drawn.
            is_in_count (bool): Whether to display the in count or out count.
        """
        text_width, text_height = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )[0]

        text_background_img = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        box_background_img = text_background_img.copy()

        text_anchor = self._get_text_anchor(
            line_counter=line_counter,
            text_width=text_width,
            text_height=text_height,
            is_in_count=is_in_count,
            centered=False,
        )

        text_img = self._draw_text(text_background_img, text, text_anchor)
        box_img = self._draw_box(
            box_background_img, text_width, text_height, text_anchor
        )

        text_img_rotated = self._rotate_img(
            text_img, line_counter, rotation_center=text_anchor
        )
        box_img_rotated = self._rotate_img(
            box_img, line_counter, rotation_center=text_anchor
        )

        frame = self._annotate_text(frame, text_img_rotated)
        frame = self._annotate_box(frame, box_img_rotated)

        return frame

    def _get_line_angle(self, line_counter: LineZone):
        """
        Calculate the line counter angle using trigonometry.

        Attributes:
            line_counter (LineZone): The line counter object used to annotate.

        Returns:
            float: Line counter angle.
        """
        start_point = line_counter.vector.start.as_xy_int_tuple()
        end_point = line_counter.vector.end.as_xy_int_tuple()

        try:
            line_angle = math.degrees(
                math.atan(
                    (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
                )
            )
            if (end_point[0] - start_point[0]) < 0:
                line_angle = 180 + line_angle
        except ZeroDivisionError:
            # Add support for vertical lines.
            line_angle = 90
            if (end_point[1] - start_point[1]) < 0:
                line_angle = 180 + line_angle

        return line_angle

    def _get_text_anchor(
        self,
        line_counter: LineZone,
        text_width: int,
        text_height: int,
        is_in_count: bool,
        centered: bool,
    ):
        """
        Set the position of the rotated image using line counter end point as reference.

        Attributes:
            line_counter (LineZone): The line counter object used to annotate.
            text_width (int): Text width.
            text_height (int): Text height.
            is_in_count (bool): Whether the text should be placed over or below the line.

        Returns:
            [int, int]: xy insertion point to place text/text-box images in frame.
        """
        end_point = line_counter.vector.end.as_xy_int_tuple()
        line_angle = self._get_line_angle(line_counter)

        # Move text anchor along and perpendicular to the line.
        text_anchor = list(end_point)

        move_along_x = int(
            math.cos(math.radians(line_angle)) * (text_width + self.text_padding)
        )
        move_along_y = int(
            math.sin(math.radians(line_angle)) * (text_width + self.text_padding)
        )

        move_perp_x = int(
            math.sin(math.radians(line_angle)) * (text_height / 2 + self.text_padding)
        )
        move_perp_y = int(
            math.cos(math.radians(line_angle)) * (text_height / 2 + self.text_padding)
        )

        text_anchor[0] -= move_along_x
        text_anchor[1] -= move_along_y
        if is_in_count:
            text_anchor[0] += move_perp_x
            text_anchor[1] -= move_perp_y
        else:
            text_anchor[0] -= move_perp_x
            text_anchor[1] += move_perp_y

        return text_anchor

    def _draw_text(
        self, text_background_img: np.ndarray, text: str, text_anchor: tuple
    ):
        """
        Draw text-box centered in the background image.

        Attributes:
            text_background_img (np.ndarray): Empty background image.
            text (str): Text to draw in the background image.
            text_anchor (int, int): xy insertion point to center text in background.

        Returns:
            np.ndarray: Background image with text drawed in it.
        """
        cv2.putText(
            text_background_img,
            text,
            text_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            (255, 255, 255),
            self.text_thickness,
            cv2.LINE_AA,
        )

        return text_background_img

    def _draw_box(
        self,
        box_background_img: np.ndarray,
        text_width: int,
        text_height: int,
        text_anchor: tuple,
    ):
        """
        Draw text-box centered in the background image.

        Attributes:
            box_background_img (np.ndarray): Empty background image.
            text_width (int): Text width.
            text_height (int): Text height.
            text_anchor (int, int): xy point to center text insertion.

        Returns:
            np.ndarray: Background image with text-box drawed in it.
        """
        box = Rect(
            x=text_anchor[0],
            y=text_anchor[1] - text_height,
            width=text_width,
            height=text_height,
        ).pad(padding=self.text_padding)

        cv2.rectangle(
            box_background_img,
            box.top_left.as_xy_int_tuple(),
            box.bottom_right.as_xy_int_tuple(),
            (255, 255, 255),
            -1,
        )

        return box_background_img

    def _rotate_img(
        self, img: np.ndarray, line_counter: LineZone, rotation_center: Tuple
    ):
        """
        Rotate img using line counter angle.

        Attributes:
            img (np.ndarray): Original image to rotate.
            line_counter (LineZone): The line counter object used to annotate.

        Returns:
            np.ndarray: Image with the same shape as input but with rotated content.
        """
        line_angle = self._get_line_angle(line_counter)

        rotation_angle = -(line_angle)
        rotation_scale = 1

        rotation_matrix = cv2.getRotationMatrix2D(
            rotation_center, rotation_angle, rotation_scale
        )

        img_rotated = cv2.warpAffine(img, rotation_matrix, (img.shape[0], img.shape[1]))

        return img_rotated

    def _annotate_box(self, frame, img):
        """
        Annotate text-box image in the original frame.

        Attributes:
            frame (np.ndarray): The base image on which to insert the text-box image.
            img (np.ndarray): text-box image.
            img_bbox (list): xyxy insertion bbox to place text-box image in frame.

        Returns:
            np.ndarray: Annotated frame.
        """
        mask = img > 95
        for i in range(3):
            frame[:, :, i][mask] = self.color.as_bgr()[i]

        return frame

    def _annotate_text(self, frame, img):
        """
        Annotate text image in the original frame.

        Attributes:
            frame (np.ndarray): The base image on which to insert the text image.
            img (np.ndarray): text image.
            img_bbox (list): xyxy insertion bbox to place text image in frame.

        Returns:
            np.ndarray: Annotated frame.
        """
        mask = img != 0
        for i in range(3):
            frame[:, :, i][mask] = self.text_color.as_bgr()[i]

        return frame
