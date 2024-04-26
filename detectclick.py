from cfg import cfg


class DetectClick():

    def __init__(self, circleCenters):
        self.circleCenters = circleCenters


    def detectClick(self, fingertip):
        curPg = cfg["currentpage"]
        buttons = cfg["pages"][curPg]["buttons"]
        radius = cfg["btnparams"]["R"]

        for i, button in enumerate(buttons):
            if button and button.strip():
                center_x = self.circleCenters[i][0]
                center_y = self.circleCenters[i][1]
                if ((fingertip[0] - center_x)**2 + (fingertip[1] - center_y)**2) < radius**2:
                    return i

class Click():

    def __init__(self, circleCenters):
        self.circleCenters = circleCenters


    def detect(self, fingertip):
        curPg = cfg["curretpage"]
        buttons = cfg["pages"][curPg]["buttons"]
        radius = cfg["btnparams"]["R"]

        for i, button in enumerate(buttons):
            if button and button.strip():
                center_x = self.circleCenters[i][0]
                center_y = self.circleCenters[i][1]
                if ((fingertip[0] - center_x)**2 + (fingertip[1] - center_y)**2) < radius**2:
                    return i

class Amt():

    def __init__(self, circleCenters):
        self.circleCenters = circleCenters


    def detectkey(self, fingertip):
        curPg = cfg["curpage"]
        buttons = cfg["pages"][curPg]["buttons"]
        radius = cfg["btnparams"]["R"]

        for i, button in enumerate(buttons):
            if button and button.strip():
                center_x = self.circleCenters[i][0]
                center_y = self.circleCenters[i][1]
                if ((fingertip[0] - center_x)**2 + (fingertip[1] - center_y)**2) < radius**2:
                    return i


