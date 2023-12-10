import os

class Obj_2_Lum_txt:

    def __init__(self,input_name,output_name,folder,image_format="png",nb_digit = 4) -> None:
        self.input_name_file =input_name
        self.input_file = open(input_name,"r")
        self.output_name_file = output_name
        self.output_file = open(output_name,"w")
        self.folder_image = folder
        self.suffixe_image = "." + image_format
        self.light_num = 0
        self.nb_digit = nb_digit

    def __del__(self):
        self.input_file.close()
        self.output_file.close()
    def read_line(self,line) -> bool:
        split = line.split()
        if split[0] == "#" or split[0] == "o":

        # Case of classical element in obj file before the vertices (comment from Blender or name of the object at the origin)
            return True
        elif split[0] == "mtllib":
            # If mtllib has first element of line, a mtl file could maybe exist, so remove it because useless in our case 
            mtl_file_name = self.input_name_file[:-3] + "mtl"
            if os.path.isfile(mtl_file_name):
                os.remove(mtl_file_name)
                print(f"Suppression fichier mtl")
            return True
        elif split[0] == "v":
            # v for vertices, the informations that interest us
            # The axis are x, y and z (in Export obj in BLender, should choose y for forward axis and z for up axis)
            image_name = self.folder_image + f"{self.light_num:0{self.nb_digit}}" + self.suffixe_image
            self.output_file.write(f"{image_name} {split[1]} {split[2]} {split[3]}\n")
            self.light_num += 1
            return True
        else:
            # If other first element, normally should have pass all vertices and the rest are useless information
            return False

    def read_lines(self) -> None:
        for line in self.input_file.readlines():
            continue_bool = self.read_line(line)
            if not(continue_bool):
                return

def main():

    transformer = Obj_2_Lum_txt("./Torus_R&B/Geometrie_dome.obj","./lumiere_torusR&B.txt","Torus_R&B/")
    transformer.read_lines()
    del transformer


if __name__ == '__main__':
    main()    

