Applies a new color palette to an image. The image to which the new color palette is applied to is called the "base image". The image from which the new color palette is derived from is called the "palette image". The following method is applied to apply a new color palette:
1. Use K-means to find N colors that best partition all the colors in the base image into N groups.
2. Use K-means to find N colors that best partition all the colors in the palette image into N groups.
3. Find a mapping between the N colors of the base image and the N colors of the palette image that minimizes the norm of the mapping. The norm of the mapping is the sum of the norms between the colors that are mapped to each other.
4. Label all pixels in the base image to its corresponding color found via K-means in step 1.
5. Apply the mapping found in step 3 to the labeled base image from step 4 to generate the final image. 
