from image_processor import ImageProcessor

if __name__ == "__main__":

    for i in range(19):
        try:
            imgs = ImageProcessor(
                f"raw_data/{i+1}.pdf", DEBUG_MODE=True
            ).get_cell_images()
        except:
            pass
    """
    imgs = ImageProcessor(f"raw_data/7.pdf", DEBUG_MODE=True).get_cell_images()
"""
