def show_image(plt, image, title, duration=None):
    if duration is not None:
        title += ' duration: ' + str(duration)
    plt.title(title, color='red')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
