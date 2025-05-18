package Main;

import Model.Hero;
import Observer.Player;

public class Main {
	
	public static void main(String[] args) {
		Hero Joy = new Hero("Joy");
		Hero Miya = new Hero("Miya");
		
		Player Mbappe = new Player("Mbappe");
		Player Messi = new Player("Messi");
		
		Joy.addObserver(Mbappe);
		Joy.addObserver(Messi);
		Miya.addObserver(Messi);
		
		Joy.kill();
		Miya.getTurtle();
		Joy.die();
		
		System.out.println("\n ======================= History Notif ==========\n");
		
		Mbappe.showNotificationHistory();
		Messi.showNotificationHistory();
		
	}
	
}
