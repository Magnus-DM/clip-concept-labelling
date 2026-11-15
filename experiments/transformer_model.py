import torch
import torch.nn as nn

class AttributeClassifier(nn.Module):
    def __init__(self, preprocessing, input_size, regularization, dropout_rate, nhead=8, num_encoder_layers=2, dim_feedforward=2048):
        """
        Args:
            regularization (bool): Of we dropout willen toepassen.
            dropout_rate (float): Dropoutrate indien regularization True is.
            nhead (int): Het aantal attention heads in de transformer.
            num_encoder_layers (int): Aantal transformer encoder lagen.
            dim_feedforward (int): Dimensie van de feedforward lagen in de transformer encoder.
        """
        super(AttributeClassifier, self).__init__()
        self.regularization = regularization

        # Transformer Encoder laag met d_model = 512 (want elke token is 512-dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, 
                                                   nhead=nhead, 
                                                   dropout=dropout_rate,
                                                   dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Indien regularization True is, definieer een dropout laag
        if self.regularization:
            self.dropout = nn.Dropout(dropout_rate)
        
        # Lineaire laag om de gecombineerde token-representatie terug te brengen tot 1 output
        self.fc = nn.Linear(512, 1)
        
        # Sigmoid activatiefunctie zodat de output tussen 0 en 1 komt.
        self.activation = nn.Sigmoid()

    def forward(self, x, regularization):
        """
        Args:
            x (Tensor): Een tensor van shape (batch_size, 1024)
        Returns:
            Tensor: Een tensor van shape (batch_size, 1) met waarden tussen 0 en 1.
        """
        input_list = x

        x = input_list[0]
        for i in range(1, len(input_list)):
            x = torch.cat((x, input_list[i]), dim=2)
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)  #
        
        # Splits de 1024-dimensionale vector in twee tokens van elk 512 dimensies.
        token1 = x[:, :512]  # Eerste 512 waarden
        token2 = x[:, 512:] 
        
        # Maak een sequence van tokens. De transformer verwacht een input met shape (seq_len, batch_size, d_model).
        # Hier maken we een sequence van twee tokens (seq_len = 2).
        tokens = torch.stack([token1, token2], dim=0)  # Shape: (2, batch_size, 512)
        
        # Voer de tokens door de transformer encoder
        transformer_output = self.transformer_encoder(tokens)  # Output: (2, batch_size, 512)
        
        # Combineer de outputs van beide tokens. Hier kiezen we voor een gemiddelde over de sequence-dimensie.
        combined = transformer_output.mean(dim=0)  # Shape: (batch_size, 512)
        
        # Indien regularization gewenst is, pas dropout toe.
        if self.regularization:
            combined = self.dropout(combined)
        
        # Geef via een lineaire laag een tensor met 1 waarde per batch door en activeer met sigmoid.
        x_out = self.fc(combined)  # Shape: (batch_size, 1)
        x_out = self.activation(x_out)
        
        return x_out