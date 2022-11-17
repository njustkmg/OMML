import torch
import torch.nn as nn

# Fully Convolutional Network
class Fcn(nn.Module):
    def __init__(self,
                 input_dim,
                 feature_dim,
                 num_labels):
        '''
        :param input_dim:
        :param feature_dim:
        :param num_labels:
        '''
        super(Fcn, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_labels = num_labels

        self.feature_layer = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.ReLU()
        )
        self.prediction_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.num_labels),
            nn.Sigmoid()
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch, labels):
        feature = self.feature_layer(batch)
        logit = self.prediction_layer(feature)
        loss = self.criterion(logit, labels)
        return feature, logit, loss

class DOMFN(nn.Module):
    def __init__(self,
                 text_hidden_dim=None,
                 vision_hidden_dim=None,
                 audio_hidden_dim=None,
                 feature_dim=None,
                 num_labels=None,
                 fusion='concat'):
        '''
        :param text_hidden_dim:
        :param vision_hidden_dim:
        :param audio_hidden_dim:
        :param feature_dim:
        :param num_labels:
        :param fusion:
        '''

        super(DOMFN, self).__init__()
        self.text_hidden_dim = text_hidden_dim
        self.vision_hidden_dim = vision_hidden_dim
        self.audio_hidden_dim = audio_hidden_dim
        self.feature_dim = feature_dim
        self.num_labels = num_labels
        self.fusion = fusion

        self.text_encoder = Fcn(self.text_hidden_dim, self.feature_dim, self.num_labels)
        self.vision_encoder = Fcn(self.vision_hidden_dim, self.feature_dim, self.num_labels)
        self.audio_encoder = Fcn(self.audio_hidden_dim, self.feature_dim, self.num_labels)

        self.multi_encoder = nn.Sequential(
            nn.Linear((3 if self.fusion == 'concat' else 1)*self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_labels),
            nn.Sigmoid()
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,
                text_embeds=None,
                vision_embeds=None,
                audio_embeds=None,
                labels=None,
                eta=None):
        text_feat, text_logit, text_celoss = self.text_encoder(text_embeds, labels)
        vision_feat, vision_logit, vision_celoss = self.vision_encoder(vision_embeds, labels)
        audio_feat, audio_logit, audio_celoss = self.audio_encoder(audio_embeds, labels)
        multi_logit = None

        if self.fusion == 'concat':
            multi_feat = torch.cat([text_feat, vision_feat, audio_feat], dim=1)
            multi_logit = self.multi_encoder(multi_feat)
        elif self.fusion == 'mean':
            multi_feat = torch.mean(torch.stack([text_feat, vision_feat, audio_feat], dim=1), dim=1)
            multi_logit = self.multi_encoder(multi_feat)
        elif self.fusion == 'max':
            multi_feat, _ = torch.max(torch.stack([text_feat, vision_feat, audio_feat], dim=1), dim=1)
            multi_logit = self.multi_encoder(multi_feat)

        multi_celoss = self.criterion(multi_logit, labels)
        average_logit = torch.mean(torch.stack([text_logit, vision_logit, audio_logit], dim=1), dim=1)
        text_penalty, _ = torch.max((text_logit-average_logit)**2, dim=1)
        vision_penalty, _ = torch.max((vision_logit-average_logit)**2, dim=1)
        audio_penalty, _ = torch.max((audio_logit-average_logit)**2, dim=1)
        text_loss = 0.5 * text_celoss * text_celoss - eta * text_penalty
        vision_loss = 0.5 * vision_celoss * vision_celoss - eta * vision_penalty
        audio_loss = 0.5 * audio_celoss * audio_celoss - eta * audio_penalty
        multi_loss = text_celoss + vision_celoss + audio_celoss + multi_celoss

        return {
            'text_logit': text_logit,
            'vision_logit': vision_logit,
            'audio_logit': audio_logit,
            'multi_logit': multi_logit,

            'text_celoss': text_celoss,
            'vision_celoss': vision_celoss,
            'audio_celoss': audio_celoss,

            'text_penalty': text_penalty,
            'vision_penalty': vision_penalty,
            'audio_penalty': audio_penalty,

            'text_loss': text_loss,
            'vision_loss': vision_loss,
            'audio_loss': audio_loss,
            'multi_loss': multi_loss
        }













