from src.model_trainer import ModelTrainer


def main():
    trainer = ModelTrainer(input_shape=(224, 224, 3), num_classes=3)
    trainer.train(
        dataset_path='data/dataset',
        epochs=20,
        batch_size=32,
        validation_split=0.2,
    )
    trainer.save_model('data/models/custom_model.h5')
    print('Model trained and saved to data/models/custom_model.h5')


if __name__ == '__main__':
    main()

