class Status:
    def __init__(self):
        # Zone 1 (right of safety)
        self.person_in_zone_1: int = 0
        self.object_in_zone_1: int = 0

        # Zone safety (middle zone)
        self.person_in_zone_safety: int = 0
        self.object_in_zone_safety: int = 0

        # Zone 2 (left of safety)
        self.person_in_zone_2: int = 0
        self.object_in_zone_2: int = 0

    def is_presence_zone_1(self):
        return self.person_in_zone_1 > 0 or self.object_in_zone_1 > 0

    def is_presence_zone_safety(self):
        return self.person_in_zone_safety > 0 or self.object_in_zone_safety > 0

    def is_presence_zone_2(self):
        return self.person_in_zone_2 > 0 or self.object_in_zone_2 > 0

    def reset(self):
        # Zone 1 (right of safety)
        self.person_in_zone_1 = 0
        self.object_in_zone_1 = 0
        self.person_in_zone_safety = 0
        self.object_in_zone_safety = 0
        # Zone 2 (left of safety)
        self.person_in_zone_2 = 0
        self.object_in_zone_2 = 0
