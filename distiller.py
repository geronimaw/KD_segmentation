from tensorflow.keras.models import Model
from tensorflow import GradientTape
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.keras.utils.io_utils import path_to_string
import h5py


class Distiller(Model):
    def __init__(self):
        super(Distiller, self).__init__()

    def call(self, teacher, student=None, batch=None):
        self.teacher = teacher
        self.student = student
        self.batch = batch
        return self

    def compiles(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, loss=1, alpha=0.8):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.loss = loss

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher

        with GradientTape() as tape:
            teacher_fms = self.teacher(x, training=False)#[1]
            # teacher_predictions = teacher_pred[1]
            # teacher_fms = teacher_pred[0]

            # Forward pass of student
            student_pred = self.student(x, training=True)
            student_predictions = student_pred
            # student_predictions = student_pred[2]
            # student_fms = student_pred[1]

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            # distillation_loss = self.distillation_loss_fn(teacher_fms, student_fms)

            # it = int(x.shape[-1] / self.batch)
            # distillation_loss = 0
            # for i in range(it):
            #     distillation_loss += self.distillation_loss_fn(x=teacher_fms[:,:,:,i*self.batch:(i+1)*self.batch],
            #                                                    y=student_fms[:,:,:,i*self.batch:(i+1)*self.batch],
            #                                                    batch=self.batch)

            # loss = 1 - (1 - self.alpha) * student_loss + self.alpha * distillation_loss
            loss = student_loss

        # Compute gradients
        trainable_vars = self.teacher.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": 1 - student_loss, "distillation_loss": distillation_loss, "loss": loss})
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)#[2]

        # Calculate the loss
        loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results


    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self._assert_weights_created()
        filepath = path_to_string(filepath)
        filepath_is_h5 = filepath.endswith('.hdf5')
        if save_format is None:
            if filepath_is_h5:
                save_format = 'h5'
            else:
                save_format = 'tf'
        else:
            user_format = save_format.lower().strip()
            if user_format in ('tensorflow', 'tf'):
                save_format = 'tf'
            elif user_format in ('hdf5', 'h5', 'keras'):
                save_format = 'h5'
            else:
                raise ValueError(
                    'Unknown format "%s". Was expecting one of {"tf", "h5"}.' % (
                        save_format,))

        if save_format == 'h5':
            with h5py.File(filepath, 'w') as f:
                hdf5_format.save_weights_to_hdf5_group(f, self.student.layers)
