import random
from dataset import get_pet_dataloader
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


batch_size = 64


def plot_inference(ckpt_path, unet):
    
    _, test_loader = get_pet_dataloader(".", batch_size=batch_size)
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

    # Extract the model's state_dict from the checkpoint
    model_state_dict = checkpoint['state_dict']

    # Load the state_dict into your model
    unet.load_state_dict(model_state_dict)


    itera = iter(test_loader)
    for _ in range(random.randint(1, 5)):
        batch = next(itera)
        inputs, target_mask = batch
        idx = random.randint(0, batch_size)

    input_image = inputs[idx].squeeze(0).cpu().numpy()
    ground_truth_mask = target_mask[idx]
    ground_truth_mask = F.one_hot(
        ground_truth_mask, num_classes=unet.num_classes
    ).squeeze(0) * 255


    input_image_tensor = inputs[idx].unsqueeze(0)

    # Make predictions using the U-Net model
    with torch.no_grad():
        predicted_mask = unet(input_image_tensor.to("cpu"))

        
    # Convert the predicted mask tensor to a NumPy array
    predicted_mask = predicted_mask.squeeze(0).argmax(axis=0).unsqueeze(0)
    print(predicted_mask.shape)
    predicted_mask = F.one_hot(predicted_mask, num_classes=3)\
        .squeeze(0) * 255
    predicted_mask = predicted_mask.cpu().numpy()

    # Plot the results
    plt.figure(figsize=(12, 4))

    # Plot the input image
    plt.subplot(131)
    plt.imshow(input_image.transpose(1,2,0), cmap='gray')
    plt.title('Input Image')
    plt.xticks([])
    plt.yticks([])  

    plt.subplot(132)
    plt.imshow(ground_truth_mask, cmap='jet')
    plt.title('Ground Truth Mask')
    plt.xticks([])
    plt.yticks([])  

    # Plot the predicted mask
    plt.subplot(133)
    plt.imshow(predicted_mask, cmap='jet')
    plt.title('Predicted Mask')
    plt.xticks([])
    plt.yticks([])  

    plt.tight_layout()
    plt.show()


