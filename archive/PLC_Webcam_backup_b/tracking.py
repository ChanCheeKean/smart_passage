import math
from typing import List
from typing import Tuple
from typing import Dict
from typing import List
from enum import Enum


class BlobType(str, Enum):
    NONE = "NONE"
    HUMAN = "HUMAN"
    OBJECT = "OBJECT"


def get_center_of_bounding_box(bounding_box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Return center of bounding box

    Parameters:
        bounding_box: Tuple of 4 int : (x, y, w, h)
    Return:
        (x_center, y_center) Center of bounding_box
    """
    return (int(bounding_box[0] + (bounding_box[2] / 2)),
            int(bounding_box[1] + (bounding_box[3] / 2))
            )


def get_area_of_bounding_box(bounding_box: Tuple[int, int, int, int]) -> int:
    """
    Return center of bounding box

    Parameters:
        bounding_box: Tuple of 4 int : (x, y, w, h)
    Return:
        (x_center, y_center) Center of bounding_box
    """
    # area = w * h
    return bounding_box[2] * bounding_box[3]


class ElementTracked:
    def __init__(self):
        self.identification: int = 0
        ''' Bounding box coordinate (x, y, w, h) '''
        self.bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.velocity: Tuple[int, int] = (0, 0)
        self.type: BlobType = BlobType.NONE

    def get_center(self) -> Tuple[int, int]:
        return get_center_of_bounding_box(self.bounding_box)

    def get_area(self) -> int:
        return get_area_of_bounding_box(self.bounding_box)


class TrackerInfo:
    element: ElementTracked
    disappearedCounter: int

    def __init__(self, element: ElementTracked, counter: int):
        self.element = element
        self.disappearedCounter = counter


# TODO: C'est vraiment sale comme code ...
class Tracker2:
    def __init__(self, threshold: int, range_id=None):
        if range_id is None:
            range_id = [1, 100]

        self._elements: Dict[int, ElementTracked] = {}
        self._elementsDisappeared: Dict[int, TrackerInfo] = {}
        self._nextId: int = range_id[0]
        self._rangeId: List[int, int] = range_id
        self._threshold = threshold

    @property
    def elements(self) -> Dict[int, ElementTracked]:
        return self._elements

    def add_element(self, element: ElementTracked) -> None:
        element.identification = self._nextId
        if self._nextId + 1 > self._rangeId[1]:
            self._nextId = self._rangeId[0]
        else:
            self._nextId += 1
        self._elements[element.identification] = element

    def update(self, new_elements: List[ElementTracked]) -> None:
        available_element_tracked = self._elements.copy()
        available_element_list = list(available_element_tracked.values())
        # Sort by center, from left to right
        available_element_list.sort(key=lambda data: data.get_center()[0])
        new_elements.sort(key=lambda data: data.get_center()[0])

        if len(available_element_list) == len(new_elements):
            # simple case when there are same number of bounding box in new frame and element in our tracking module
            for i in range(len(available_element_list)):
                self._elements[i].bounding_box = new_elements[i].bounding_box
                if self._elements[i].type != BlobType.HUMAN:
                    self._elements[i].type = new_elements[i].type
        else:
            # get distance for all association
            association: List[Tuple[int, Tuple[ElementTracked, ElementTracked]]] = []
            for element in available_element_list:
                for new_element in new_elements:
                    distance = abs(new_element.get_center()[0] - (element.get_center()[0] + element.velocity[0]))
                    association.append((distance, (element, new_element)))

            association.sort(key=lambda data: data[0])
            while len(association) > 0 and len(available_element_list) > 0 and len(new_elements) > 0:
                (_, (element_association, new_element_association)) = association.pop(0)
                association = [data for data in association if
                               (data[1][0].get_center()[0] != element_association.get_center()[0] and
                                data[1][1].get_center()[0] != new_element_association.get_center()[0])]
                available_element_list = [element for element in available_element_list if
                                element.get_center()[0] != element_association.get_center()[0]]
                new_elements = [element for element in new_elements if
                                element.get_center()[0] != new_element_association.get_center()[0]]
                # TODO: Update elementList in tracking
                new_element_association.identification = element_association.identification
                if self._elements[element_association.identification].type == BlobType.HUMAN:
                    new_element_association.type = BlobType.HUMAN
                new_element_association.velocity = (
                    new_element_association.get_center()[0] - self._elements[element_association.identification].get_center()[0],
                    new_element_association.get_center()[1] - self._elements[element_association.identification].get_center()[1]
                )
                self._elements[element_association.identification] = new_element_association

            if len(association) == 0:
                print("There is a problem in this run tracking")
            if len(available_element_list) != 0 and len(new_elements) != 0:
                print("There is a problem in this run tracking, both lists are not empty")
            elif len(new_elements) > 0:  # len(available_element_list) == 0
                # See if cannot be disappeared element
                # get distance for all association
                association: List[Tuple[int, Tuple[int, int]]] = []
                for _, (_, element) in self._elementsDisappeared.items():
                    for new_element in new_elements:
                        distance = abs(new_element.get_center()[0] - (element.get_center()[0] + element.velocity[0]))
                        association.append((distance, (element.identification, new_element.identification)))

                association.sort(key=lambda data: data[0])
                while len(association) > 0 and len(self._elementsDisappeared) > 0 and len(new_elements) > 0:
                    (distance, (element_association, new_element_association)) = association.pop(0)
                    if distance > self._threshold:
                        break
                    else:
                        association = [data for data in association if
                                       (data[1][0].get_center()[0] != element_association.get_center()[0] and
                                        data[1][1].get_center()[0] != new_element_association.get_center()[0])]
                        new_elements = [element for element in new_elements if
                                        element.get_center()[0] != new_element_association.get_center()[0]]
                        del self._elementsDisappeared[element_association.identification]
                        new_element_association.identification = element_association.identification
                        if element_association.type == BlobType.HUMAN:
                            new_element_association.type = BlobType.HUMAN
                        self._elements[element_association.identification] = new_element_association
                if len(new_elements) > 0:
                    for new_element in new_elements:
                        self.add_element(new_element)
            else:  # len(new_elements) == 0 and len(available_element_list) > 0
                # Delete passengers because disappeared
                for element in available_element_list:
                    self._elementsDisappeared[element.identification] = TrackerInfo(element, 0)

            if self._elementsDisappeared:
                element_to_delete = []
                for id_element, elementTrackerInfo in self._elementsDisappeared.items():
                    element = elementTrackerInfo.element
                    counter = elementTrackerInfo.disappearedCounter
                    if counter < 10:
                        self._elementsDisappeared[id_element] = TrackerInfo(element, counter + 1)
                    else:
                        element_to_delete.append(id_element)
                for element in element_to_delete:
                    del self._elementsDisappeared[element]
            

class Tracker:
    def __init__(self, threshold: int, range_id=None):
        if range_id is None:
            range_id = [1, 100]

        self._elements: Dict[int, ElementTracked] = {}
        self._elementsDisappeared: Dict[int, TrackerInfo] = {}
        self._threshold: int = threshold
        self._nextId: int = range_id[0]
        self._rangeId: List[int, int] = range_id

    @property
    def elements(self) -> Dict[int, ElementTracked]:
        return self._elements

    def add_element(self, element: ElementTracked) -> None:
        element.identification = self._nextId
        if self._nextId + 1 > self._rangeId[1]:
            self._nextId = self._rangeId[0]
        else:
            self._nextId += 1
        self._elements[element.identification] = element

    def update(self, new_elements: List[ElementTracked]) -> None:
        available_element_tracked = self._elements.copy()
        for new_element in new_elements:
            # Center of contour
            new_element_center = get_center_of_bounding_box(new_element.bounding_box)

            min_distance = self._threshold
            min_distance_id: int = -1

            for id_element, element in available_element_tracked.items():
                element_center = element.get_center()
                # Get distance
                distance = abs(element_center[0] - new_element_center[0])
                print("id element {} -> id test {} distance = {}".format(new_element.identification,
                                                                         element.identification, distance))

                '''
                w_new_element = new_element.bounding_box[2]
                h_new_element = new_element.bounding_box[3]
                w_element = element.bounding_box[2]
                h_element = element.bounding_box[3]
                distance += 100 * ((abs(w_new_element - w_element) + abs(h_new_element - h_element)) / (max(w_new_element, w_element) + (max(h_new_element, h_element))))
                '''
                '''
                print("id element {} -> id test {} distance without area = {}".format(new_element.identification, element.identification, distance))
                # Mettre sous racine ?
                distance = distance + abs(element.get_area() - new_element.get_area())
                print("id element {} -> id test {} distance with area = {}".format(new_element.identification, element.identification, distance))
                '''

                if distance < min_distance:
                    min_distance = distance
                    min_distance_id = id_element

            # Found a candidate
            if min_distance_id != -1:
                new_element.identification = min_distance_id
                if self._elements[min_distance_id].type == BlobType.HUMAN:
                    new_element.type = BlobType.HUMAN
                self._elements[min_distance_id] = new_element
                del available_element_tracked[min_distance_id]
            else:
                for id_element, elementTrackerInfo in self._elementsDisappeared.items():
                    element = elementTrackerInfo.element
                    element_center = element.get_center()
                    # Get distance
                    distance = abs(element_center[0] - new_element_center[0])

                    if distance < min_distance:
                        min_distance = distance
                        min_distance_id = id_element

                if min_distance_id != -1:
                    new_element.identification = min_distance_id
                    if self._elementsDisappeared[min_distance_id].element.type == BlobType.HUMAN:
                        new_element.type = BlobType.HUMAN
                    self._elements[min_distance_id] = new_element
                    del self._elementsDisappeared[min_distance_id]
                else:
                    # New element
                    self.add_element(new_element)

        if self._elementsDisappeared:
            element_to_delete = []
            for id_element, elementTrackerInfo in self._elementsDisappeared.items():
                element = elementTrackerInfo.element
                counter = elementTrackerInfo.disappearedCounter
                if counter < 5:
                    self._elementsDisappeared[id_element] = TrackerInfo(element, counter + 1)
                else:
                    element_to_delete.append(id_element)
            for element in element_to_delete:
                del self._elementsDisappeared[element]

        if available_element_tracked:
            for id_element, element in available_element_tracked.items():
                self._elementsDisappeared[id_element] = TrackerInfo(element, 0)
                del self._elements[id_element]
