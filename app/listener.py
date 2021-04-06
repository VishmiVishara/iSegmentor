from views import reciver

class AB:

    re = reciver()

    def call(self, epoch, train_discriminator_loss_meter,
                        train_generator_loss_meter,
                        train_pixel_loss,
                        train_adversarial_loss_meter,
                        pixAcc,
                        mIoU):
        self.re.call(epoch, train_discriminator_loss_meter,
                        train_generator_loss_meter,
                        train_pixel_loss,
                        train_adversarial_loss_meter,
                        pixAcc,
                        mIoU)
