from fastapi import APIRouter
import base64

router = APIRouter()

@router.get("/overview")
async def overview():
    example_image = "data/example_image.jpg"  # Chemin d'un exemple local
    example_mask = "data/example_mask.png"  # Chemin du masque généré
    
    with open(example_image, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode()
    with open(example_mask, "rb") as mask_file:
        encoded_mask = base64.b64encode(mask_file.read()).decode()
    
    return {
        "example_input": f"data:image/jpeg;base64,{encoded_image}",
        "example_prediction": f"data:image/png;base64,{encoded_mask}"
    }
