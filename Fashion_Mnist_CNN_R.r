# Setting up environment:

install.packages("keras")
install.packages("ggplot2")
install.packages("dplyr")
library(keras)
install_k

# Load Dataset:

fashion_mnist <- dataset_fashion_mnist()
train_images <- fashion_mnist$train$x
train_labels <- fashion_mnist$train$y
test_images <- fashion_mnist$test$x
test_labels <- fashion_mnist$test$y

library(ggplot2)
library(dplyr)

# Display the shape of the dataset
dim(train_images)
dim(test_images)

# Display the first image and its label
image(matrix(train_images[1,,], nrow=28, ncol=28), col=gray.colors(256), main=paste("Label:", train_labels[1]))

# Normalization:

train_images <- train_images / 255
test_images <- test_images / 255
# Reshape the data to add a channel dimension
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))

# Building CNN Model:

model <- keras_model_sequential() %>%
layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
layer_max_pooling_2d(pool_size = c(2, 2)) %>%
layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
layer_flatten() %>%
layer_dense(units = 64, activation = 'relu') %>%
layer_dense(units = 10, activation = 'softmax')


# Compiling the model:

model %>% compile( optimizer = 'adam',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = c('accuracy')
) 

# Model training:

model %>% fit( train_images, 
              train_labels,
              epochs = 10, 
              validation_data = list(test_images, test_labels)
)

# Prediction:

predictions <- model %>% predict(test_images[1:2, , , drop = FALSE])
print(predictions)
