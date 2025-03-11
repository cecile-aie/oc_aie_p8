from fastapi import APIRouter
import base64
from app.utils.postprocessing import encode_image, encode_colored_mask

router = APIRouter()

@router.get("/overview")
async def overview():
    example_image = "data/example_image.jpg"  # Chemin d'un exemple local
    example_mask = "data/example_mask.png"  # Chemin du masque généré
    
    with open(example_image, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode()
    
    mask_array = None
    with open(example_mask, "rb") as mask_file:
        mask_array = np.array(Image.open(mask_file))
    
    encoded_colored_mask = encode_colored_mask(mask_array)
    
    return {
        "example_input": f"data:image/jpeg;base64,{encoded_image}",
        "example_prediction": f"data:image/png;base64,{encoded_colored_mask}"
    }

