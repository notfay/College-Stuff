package Observer;

import java.util.ArrayList;
import java.util.List;

public class Player implements Observer {

	private String playerName;
	private List<String> notificationHistory;

	public Player(String playerName) {
		this.playerName = playerName;
		this.notificationHistory = new ArrayList<>();
	}

	@Override
	public void update(String message) {
		// TODO Auto-generated method stub
		System.out.println( playerName + " Menerima update: " + message);
		notificationHistory.add(message);  
	}

	public void showNotificationHistory() {
		System.out.println("Notif history untuk player- " + playerName + ": ");
		for (String notification : notificationHistory) {
			System.out.println(" = " + notification);
		}
	}

	public String getPlayerName() {
		return playerName;
	}

}
